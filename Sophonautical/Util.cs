using System;
using System.IO;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;

namespace Sophonautical
{
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

        public static float[,,] AsImage(float[] data)
        {
            int BlockDim = (int)Math.Sqrt(data.Length / 3);
            var image = new float[BlockDim, BlockDim, 3];

            int i = 0;
            for (int x = 0; x < BlockDim; x++)
            for (int y = 0; y < BlockDim; y++)
            for (int c = 0; c < 3; c++)
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
