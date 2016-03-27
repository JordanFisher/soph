using System;
using System.IO;
using System.Diagnostics;

using static Sophonautical.Util;

namespace Sophonautical
{
    class Program
    {
        const string
            OutputDir = "C://Users//Jordan//Desktop//output",
            TrainingDataDir = "C://Users//Jordan//Desktop//cifar-10-batches-bin",
            LabelFile = "batches.meta.txt",
            FileName = "data_batch_1.bin";

        static Plot pl = new Plot(OutputDir);
        static Random rnd = new Random(1);

        const int
            Width = 32, Height = 32,
            Rows = 10000,
            RowSize = 3073,

            BlockDim = 5;

        const float
            GroupThreshold = .05f,
            ApplyThreshold = 10 * GroupThreshold;

        static void Test()
        {
            return;
            Console.WriteLine(s(new float[,] { { 1, 2 }, { 4, 5 } }));

            var k = new AffineKernel(new float[] { 1, 2, 3 }, new float[] { 0, 0, 0 });
            var input = new float[] { 2.1f, 4.1f, 6.1f };

            Normalize(k.w);
            var d = k.Apply(input);
            Console.WriteLine(d);

            var m = k.ArgMinInverseError(input);
            Console.WriteLine(m);

            var result = Min(x => (float)Math.Cos(x), threshold: .000001f);
            Console.WriteLine(result);

            Console.ReadLine();
        }

        static void Main(string[] args)
        {
            Test();

            // Read label meta file.
            var LabelNames = File.ReadAllLines(Path.Combine(TrainingDataDir, LabelFile));

            // Read training data.
            var bytes = File.ReadAllBytes(Path.Combine(TrainingDataDir, FileName));
            Debug.Assert(bytes.Length == Rows * RowSize);

            // Unpack training data.
            var images = new float[Rows][,,];
            var labels = new byte[Rows];
            var source_index = 0;

            for (int row = 0; row < Rows; row++)
            {
                var image = images[row] = new float[Width, Height, 3];

                labels[row] = bytes[source_index++];

                for (int channel = 0; channel < 3; channel++)
                for (int x = 0; x < Width; x++)
                for (int y = 0; y < Height; y++)
                {
                    image[x, y, channel] = bytes[source_index++] / 255f;
                    //image[x, y, channel] = bytes[source_index++];
                }
            }

            Debug.Assert(source_index == Rows * RowSize);

            var kernels = LearnLevel(images, Width, Height, 3, NumKernels: 3);
            var output = LevelOutput(kernels, images, Width, Height, 3, NumKernels: 3);
        }

        static Kernel[] LearnLevel(float[][,,] images, int Width, int Height, int Channels, int NumKernels = 3)
        {
            int BlockSize = BlockDim * BlockDim * Channels;

            // Unpack blocks.
            var blocks = new float[Rows * (Width - (BlockDim - 1)) * (Height - (BlockDim - 1))][];
            int block_index = 0;

            for (int row = 0; row < Rows; row++)
            {
                var image = images[row];

                for (int x = 0; x < Width - (BlockDim - 1); x++)
                for (int y = 0; y < Height - (BlockDim - 1); y++)
                {
                    var block = blocks[block_index++] = new float[BlockSize];

                    int i = 0;
                    for (int _x = x; _x < x + BlockDim; _x++)
                    for (int _y = y; _y < y + BlockDim; _y++)
                    for (int _c = 0; _c < Channels; _c++)
                    {
                        block[i++] = (float)image[_x, _y, _c];
                    }
                }
            }

            Debug.Assert(block_index == blocks.Length);

            // Find level 1 kernels
            var Kernels = new AffineKernel[NumKernels];
            for (int i = 0; i < NumKernels; i++)
            {
                var kernel = Kernels[i] = FindKernel(blocks, BlockSize);
                //kernel.Plot();

                var remainder = SetRemainder(blocks, kernel, GroupThreshold);
                blocks = remainder;
            }

            return Kernels;
        }

        static float[][,,] LevelOutput(Kernel[] Kernels, float[][,,] images, int Width, int Height, int Channels, int NumKernels = 3)
        {
            int BlockSize = BlockDim * BlockDim * Channels;

            // Construct level 1 output
            var _images = new float[Rows][,,];
            var _image_dim = Width - (BlockDim - 1);
            var input = new float[BlockSize];

            for (int row = 0; row < Rows; row++)
            {
                var image = images[row];
                var _image = new float[_image_dim, _image_dim, NumKernels];

                for (int x = 0; x < Width - (BlockDim - 1); x++)
                for (int y = 0; y < Height - (BlockDim - 1); y++)
                {
                    int i = 0;
                    for (int _x = x; _x < x + BlockDim; _x++)
                    for (int _y = y; _y < y + BlockDim; _y++)
                    for (int _c = 0; _c < Channels; _c++)
                    {
                        input[i++] = (float)image[_x, _y, _c];
                    }

                    for (int k = 0; k < NumKernels; k++)
                    {
                        var kernel = Kernels[k];

                        if (kernel.Score(input) < ApplyThreshold)
                        {
                            _image[x, y, k] = kernel.Apply(input);
                            input = kernel.Remainder(input);
                        }
                        else
                        {
                            _image[x, y, k] = 0;
                        }
                    }
                }

                // Side by side image comparison of input and output.
                pl.plot(image, _image);
            }

            return _images;
        }

        static float TestKernel(float[][] blocks, Kernel kernel, float threshold, int step = 1)
        {
            int count = 0, total = 0;
            for (int i = rnd.Next(step); i < blocks.Length; i += step)
            {
                total++;

                var block = blocks[i];

                float error = kernel.Score(block);
                if (error < threshold) count++;
            }

            float ratio = count / (float)total;

            //Console.WriteLine($"Kernel result: {100 * ratio}% above {threshold} threshold.");

            return ratio;
        }

        static float[][] SetRemainder(float[][] blocks, Kernel kernel, float threshold)
        {
            bool[] in_remainder = new bool[blocks.Length];

            int count = 0;
            for (int i = 0; i < blocks.Length; i++)
            {
                var block = blocks[i];

                float error = kernel.Score(block);
                if (error > threshold)
                {
                    in_remainder[i] = true;
                    count++;
                }
                else
                {
                    in_remainder[i] = false;
                }
            }

            var remainder = new float[count][];
            count = 0;
            for (int i = 0; i < blocks.Length; i++)
            {
                if (in_remainder[i])
                {
                    remainder[count] = kernel.Remainder(blocks[i]);
                    count++;
                }
            }

            Debug.Assert(remainder.Length == count);
            float percent = 100 * remainder.Length / (float)blocks.Length;
            Console.WriteLine(
                $"\nRemainder: {remainder.Length} items out of {blocks.Length}, {percent}%.\n"
            );

            return remainder;
        }

        abstract class Kernel
        {
            public virtual float Score(float[] input)
            {
                var remainder = Remainder(input);

                //var error = NormInf(remainder);
                //var error = Norm1(remainder) / input.Length;
                var error = Norm1Thresh(remainder) / input.Length;

                return error;
            }

            public virtual float[] Reconstruct(float[] input)
            {
                return Inverse(Apply(input));
            }

            public virtual float[] Remainder(float[] input)
            {
                return minus(input, Reconstruct(input));
            }

            public abstract float ArgMinInverseError(float[] input);
            public abstract float Apply(float[] input);
            public abstract float[] Inverse(float y);
            public abstract void Plot();
        }

        class AffineKernel : Kernel
        {
            public float[] w, b;

            public AffineKernel(float[] w, float[] b=null) { this.w = w; this.b = b; }

            public AffineKernel(int BlockSize)
            {
                w = new float[BlockSize];
                b = new float[BlockSize];
            }

            public override float Apply(float[] input)
            {
                return Dot(w, minus(input, b));
                //return ArgMinInverseError(input);
            }

            public override float[] Inverse(float y)
            {
                return plus(times(y, w), b);
            }

            public override float ArgMinInverseError(float[] input)
            {
                float min_a = 0, max_a = 0;
                for (int i = 0; i < input.Length; i++)
                {
                    float a = (input[i] - b[i]) / w[i];
                    if (i == 0 || a > max_a) max_a = a;
                    if (i == 0 || a < min_a) min_a = a;
                }

                //Console.WriteLine($"min = {min_a} max = {max_a}");

                var min = Min(y => Norm0(minus(input, Inverse(y))),
                    x1: min_a, x2: max_a,
                    min_x: min_a, max_x: max_a,
                    threshold: .00001f, n: 100, max_iterations: 1
                );

                return min.Item1;
            }

            public override void Plot()
            {
                pl.plot(AsImage(w), AsImage(b));
            }

            public override string ToString()
            {
                return $"{s(w)},{s(b)}";
            }
        }

        static AffineKernel FindKernel(float[][] blocks, int BlockSize)
        {
            float best_score = 0;
            AffineKernel best_kernel = null;
            for (int i = 0; i < 500; i++)
            {
                var kernel = new AffineKernel(BlockSize);

                int w_index = rnd.Next(0, blocks.Length - 1);
                int b_index = rnd.Next(0, blocks.Length - 1);
                for (int j = 0; j < BlockSize; j++)
                {
                    //kernel.w[j] = (float)rnd.NextDouble();
                    //kernel.b[j] = (float)rnd.NextDouble();

                    kernel.w[j] = blocks[w_index][j] - blocks[b_index][j];
                    kernel.b[j] = blocks[b_index][j];

                    //kernel.w[j] = blocks[w_index][j];
                    //kernel.b[j] = 0;
                }

                Normalize(kernel.w);

                var score = TestKernel(blocks, kernel, GroupThreshold, step : 599);
                if (score > best_score)
                {
                    best_score = score;
                    best_kernel = kernel;

                    Console.WriteLine($"New best {s(best_kernel.w)},{s(best_kernel.b)} with a score of {best_score}.");
                }

                //Console.WriteLine($"Iteration {i}: {best_score}, {score}");
            }

            return best_kernel;
        }
    }
}
