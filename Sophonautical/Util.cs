using System;
using System.IO;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Collections.Generic;

namespace Sophonautical
{
    public static class UtilTest
    {
        public static float[,,] TestImage()
        {
            int dim = 2;
            int channels = 2;
            var image = new float[dim, dim, channels];
            int count = 0;

            for (int x = 0; x < dim; x++)
            for (int y = 0; y < dim; y++)
            for (int c = 0; c < channels; c++)
            {
                image[x, y, c] = count++;
            }

            return image;
        }

        public static void Run()
        {
            var test_image = TestImage();

            var block = Util.AsBlock(test_image, BlockDim: 2, Channels: 2);
            var reconstruction = Util.AsImage(block, channels: 2);

            Debug.Assert(Util.Equal(test_image, reconstruction));

            var block2 = Util.AsBlock(TestImage(), BlockDim: 2, Channels: 2);

            var diff = Util.FuzzyDiff(block, block2, channels: 2);
            Debug.Assert(Util.Norm1(diff) == 0);

            block[0] = 4;
            diff = Util.FuzzyDiff(block, block2, channels: 2);
            Debug.Assert(Util.Norm1(diff) == 0);

            block[0] = 10;
            diff = Util.FuzzyDiff(block, block2, channels: 2);
            Debug.Assert(Util.Norm1(diff) == 6);
        }
    }

    class Vector
    {
        public float[] vals;

        public Vector(float[] vals)
        {
            this.vals = vals;
        }

        public static implicit operator Vector(float[] vals)
        {
            return new Vector(vals);
        }

        public static implicit operator float[] (Vector vec)
        {
            return vec.vals;
        }

        public float this[int i]
        {
            get { return vals[i]; }
            set { vals[i] = value; }
        }

        public static Vector operator +(Vector x, Vector y) { return Util.plus(x.vals, y.vals); }
        public static Vector operator -(Vector x, Vector y) { return Util.minus(x.vals, y.vals); }
        public static Vector operator *(Vector x, float y) { return Util.times(x.vals, y); }
        public static Vector operator *(float y, Vector x) { return Util.times(x.vals, y); }
        public static Vector operator /(Vector x, float y) { return Util.scalar_div(x.vals, y); }
    }

    public partial class Util
    {
        public static Plot pl = new Plot("");
        public static Random rnd = new Random(1);

        public const float eps = .00001f;

        public static void Timeit(Action a, string description)
        {
            Stopwatch stopWatch = new Stopwatch();
            stopWatch.Start();

            for (int i = 0; i < 100; i++) a();

            stopWatch.Stop();
            TimeSpan ts = stopWatch.Elapsed;

            Console.WriteLine($"{description}, took {ts.Milliseconds / 100f} ms.");
        }

        public static string s<T>(T[] data) { return pl.py_array(data); }
        public static string s(float[,] data) { return pl.py_array(data); }
        public static string s(float[,,] data) { return pl.py_array(data); }

        public static bool Equal(float[,,] a, float[,,] b)
        {
            int dimx = a.GetLength(0);
            int dimy = a.GetLength(1);
            int dimz = a.GetLength(2);

            for (int x = 0; x < dimx; x++)
            for (int y = 0; y < dimy; y++)
            for (int z = 0; z < dimz; z++)
            {
                if (a[x, y, z] != b[x, y, z]) return false;
            }

            return true;
        }

        public static float[,,] AsImage(float[] data, int channels = 3)
        {
            int BlockDim = (int)Math.Sqrt(data.Length / channels);
            var image = new float[BlockDim, BlockDim, channels];

            int i = 0;
            for (int x = 0; x < BlockDim; x++)
            for (int y = 0; y < BlockDim; y++)
            for (int c = 0; c < channels; c++)
            {
                image[x, y, c] = data[i++];
            }

            return image;
        }

        public static float[] minus(float[] x, float[] y)
        {
            var result = new float[x.Length];
            for (int i = 0; i < x.Length; i++) result[i] = x[i] - y[i];

            return result;
        }

        public static float[] plus(float[] x, float[] y)
        {
            var result = new float[x.Length];
            for (int i = 0; i < x.Length; i++) result[i] = x[i] + y[i];

            return result;
        }

        public static float[] times(float[] x, float y)
        {
            var result = new float[x.Length];
            for (int i = 0; i < x.Length; i++) result[i] = x[i] * y;

            return result;
        }

        public static float[] times(float y, float[] x)
        {
            var result = new float[x.Length];
            for (int i = 0; i < x.Length; i++) result[i] = x[i] * y;

            return result;
        }

        public static float[] FuzzyDiff(float[] block1, float[] block2, int channels)
        {
            int BlockDim = (int)Math.Sqrt(block1.Length / channels);
            var _block1 = AsImage(block1, channels: channels);
            var _block2 = AsImage(block2, channels: channels);

            for (int x = 0; x < BlockDim; x++)
            for (int y = 0; y < BlockDim; y++)
            for (int c = 0; c < channels; c++)
            {
                float                 min =               Math.Abs(_block1[x, y, c] - _block2[x, y, c]);
                if (x > 0)            min = Math.Min(min, Math.Abs(_block1[x, y, c] - _block2[x - 1, y, c]));
                if (x + 1 < BlockDim) min = Math.Min(min, Math.Abs(_block1[x, y, c] - _block2[x + 1, y, c]));
                if (y > 0)            min = Math.Min(min, Math.Abs(_block1[x, y, c] - _block2[x, y - 1, c]));
                if (y + 1 < BlockDim) min = Math.Min(min, Math.Abs(_block1[x, y, c] - _block2[x, y + 1, c]));

                _block1[x, y, c] = min;
            }

            var block = AsBlock(_block1, BlockDim: BlockDim, Channels: channels);

            return block;
        }

        public static float[] AsBlock(float[,,] image, int BlockDim, int Channels = 3)
        {
            int BlockSize = BlockDim * BlockDim * Channels;
            var block = new float[BlockSize];

            int i = 0;
            for (int _x = 0; _x < BlockDim; _x++)
            for (int _y = 0; _y < BlockDim; _y++)
            for (int _c = 0; _c < Channels; _c++)
            {
                block[i++] = (float)image[_x, _y, _c];
            }

            Debug.Assert(i == BlockSize);

            return block;
        }

        public static float NormInf(float[] vec)
        {
            float max = 0;
            for (int i = 0; i < vec.Length; i++)
            {
                max = Math.Max(max, Math.Abs(vec[i]));
            }

            return max;
        }

        public static float Norm0(float[] vec, float threshold = .1f)
        {
            float sum = 0;
            for (int i = 0; i < vec.Length; i++)
            {
                //if (Math.Abs(vec[i]) > threshold) sum++;
                //if (Math.Abs(vec[i]) > threshold) sum += Math.Abs(vec[i]) - threshold;
                sum += Math.Abs(vec[i]) > threshold ? 1 : Math.Abs(vec[i]) / threshold;
            }

            return sum;
        }

        public static float Norm1(float[] vec)
        {
            float sum = 0;
            for (int i = 0; i < vec.Length; i++)
            {
                sum += Math.Abs(vec[i]);
            }

            return sum;
        }

        public static float Norm1Thresh(float[] vec, float threshold = .2f)
        {
            float sum = 0;
            for (int i = 0; i < vec.Length; i++)
            {
                sum += Math.Abs(vec[i]) > threshold ? threshold : Math.Abs(vec[i]);
            }

            return sum;
        }

        public static float SquaredNorm2(float[] vec)
        {
            float norm = 0;
            for (int i = 0; i < vec.Length; i++)
            {
                norm += vec[i] * vec[i];
            }

            return norm;
        }

        public static float Norm2(float[] vec)
        {
            return (float)Math.Sqrt(SquaredNorm2(vec));
        }

        public static void Normalize(float[] vec)
        {
            float norm = Norm2(vec);
            for (int i = 0; i < vec.Length; i++)
            {
                vec[i] /= (norm + .00001f);
            }
        }

        public static float Dot(float[] w, float[] v)
        {
            float result = 0;
            for (int i = 0; i < w.Length; i++)
                result += w[i] * v[i];

            return result;
        }

        public static float[] scalar_div(float[] w, float v)
        {
            for (int i = 0; i < w.Length; i++)
                w[i] /= v;

            return w;
        }

        public static float Similiarty(float[] w, float[] v)
        {
            float dot = Dot(w, v);
            return Math.Abs(dot) / Norm2(v);
        }

        public static Tuple<float, float> Min(Func<float, float> f,
            float x1 = -1, float x2 = 1,
            float min_x = -1000000, float max_x = 1000000,
            int n = 5, float threshold = .05f, int max_iterations = 10
        )
        {
            int iteration = 0;
            float step = (x2 - x1) / (n - 1);
            float argmin = 0, min = 0;

            do
            {
                //Console.WriteLine(new Tuple<float, float>(argmin, min));

                int min_index = -1;
                for (int i = 0; i < n; i++)
                {
                    float x = x1 + i * step;
                    float y = f(x);

                    if (y < min || min_index < 0)
                    {
                        min = y;
                        argmin = x;
                        min_index = i;
                    }
                }

                if (min_index < 0) break;

                if (min_index != 0 && min_index != n - 1 ||
                    argmin < min_x + eps || argmin > max_x + eps)
                {
                    step *= .7f;
                }

                float spread = step * (n - 1);
                x1 = Math.Max(min_x, argmin - spread / 2f);
                x2 = Math.Min(max_x, argmin + spread / 2f);
            }
            while (step > threshold && ++iteration < max_iterations);

            return new Tuple<float, float>(argmin, min);
        }

        public static void NormalizeComponents(float[][,,] inputs, int rows, int width, int height, int channels)
        {
            float[] min = new float[channels];
            float[] max = new float[channels];
            for (int i = 0; i < channels; i++) min[i] = max[i] = inputs[0][0, 0, i];

            for (int i = 0; i < rows; i++)
            for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++)
            for (int c = 0; c < channels; c++)
            {
                min[c] = Math.Min(min[c], inputs[i][x, y, c]);
                max[c] = Math.Max(max[c], inputs[i][x, y, c]);
            }

            for (int i = 0; i < rows; i++)
            for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++)
            for (int c = 0; c < channels; c++)
            {
                inputs[i][x, y, c] = (inputs[i][x, y, c] - min[c]) / (max[c] - min[c]);
            }
        }

        public static float[] Average(List<float[]> arrays)
        {
            int dim = arrays[0].Length;
            var avg = new float[dim];

            foreach (var array in arrays)
            {
                for (int i = 0; i < dim; i++)
                {
                    avg[i] += array[i];
                }
            }

            for (int i = 0; i < dim; i++)
            {
                avg[i] /= arrays.Count;
            }

            return avg;
        }

        public static void SaveImage(float[,,] image, string name, string dir = "")
        {
            int Width = image.GetLength(0);
            int Height = image.GetLength(1);

            var _image = new byte[Width, Height, 3];

            for (int x = 0; x < Width; x++)
            for (int y = 0; y < Height; y++)
            for (int c = 0; c < 3; c++)
                _image[x, y, c] = (byte)image[x, y, c];

            SaveImage(_image, name, dir);
        }

        public static void SaveImage(byte[,,] image, string name, string dir="")
        {
            int Width = image.GetLength(0);
            int Height = image.GetLength(1);

            var bmp = new Bitmap(Width, Height);

            for (int x = 0; x < Width; x++)
            for (int y = 0; y < Height; y++)
            {
                int r = image[x, y, 0];
                int g = image[x, y, 1];
                int b = image[x, y, 2];

                bmp.SetPixel(x, y, Color.FromArgb(r, g, b));
            }

            bmp.Save(Path.Combine(dir, name + ".png"), ImageFormat.Png);
        }
    }
}
