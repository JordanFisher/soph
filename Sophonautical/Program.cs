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

        static Plot pl = new Plot(OutputDir, Quiet: false, Save: false);
        //static Plot pl = new Plot(OutputDir, Quiet: true, Save: true);
        static Random rnd = new Random(1);

        const int
            Width = 32, Height = 32,
            Rows = 10000,
            RowSize = 3073,
            NumLabels = 10;

        const float
            GroupThreshold = .075f,
            ApplyThreshold = 10 * GroupThreshold;

        static void Test()
        {
            pl.Test();
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

            Console.WriteLine("\n\nLearning level 1\n\n");

            int block_dim1 = 5, num_kernels1 = 12;
            var output_dim1 = Width - (block_dim1 - 1);

            var kernels1 = LearnLevel(images, Width, Height, 3,
                NumKernels: num_kernels1, BlockDim: block_dim1, labels: labels);
            var output1 = LevelOutput(kernels1, images, Width, Height, 3,
                NumKernels: num_kernels1, BlockDim: block_dim1);
            NormalizeComponents(output1, Rows, output_dim1, output_dim1, num_kernels1);

            Console.WriteLine("\n\nLearning level 2\n\n");

            int block_dim2 = 3, num_kernels2 = 6;
            var output_dim2 = output_dim1 - (block_dim2 - 1);

            var kernels2 = LearnLevel(output1, output_dim1, output_dim1, num_kernels1,
                NumKernels: num_kernels2, BlockDim: block_dim2, labels: labels);
            var output2 = LevelOutput(kernels2, output1, output_dim1, output_dim1, num_kernels1,
                NumKernels: num_kernels2, BlockDim: block_dim2);
            NormalizeComponents(output2, Rows, output_dim2, output_dim2, num_kernels2);

            Console.WriteLine("\n\nLearning level 3\n\n");

            int block_dim3 = 3, num_kernels3 = 3;
            var output_dim3 = output_dim2 - (block_dim3 - 1);

            var kernels3 = LearnLevel(output2, output_dim2, output_dim2, num_kernels2,
                NumKernels: num_kernels3, BlockDim: block_dim3, labels: labels);
            var output3 = LevelOutput(kernels3, output2, output_dim2, output_dim2, num_kernels2,
                NumKernels: num_kernels3, BlockDim: block_dim3);
            NormalizeComponents(output3, Rows, output_dim3, output_dim3, num_kernels3);

            // Side by side image comparison of input and output.
            for (int i = 0; i < Rows; i++)
            {
                pl.plot(images[i], output1[i], output2[i], output3[i]);
            }

            Console.ReadLine();
        }

        static Kernel[] LearnLevel(float[][,,] input, int Width, int Height, int Channels,
            int NumKernels = 3, int BlockDim = 3,
            byte[] labels = null)
        {
            int BlockSize = BlockDim * BlockDim * Channels;

            // Unpack blocks.
            int num_blocks = Rows * (Width - (BlockDim - 1)) * (Height - (BlockDim - 1));
            var blocks = new float[num_blocks][];
            var block_labels = new byte[num_blocks];
            int block_index = 0;

            for (int row = 0; row < Rows; row++)
            {
                var image = input[row];

                for (int x = 0; x < Width - (BlockDim - 1); x++)
                for (int y = 0; y < Height - (BlockDim - 1); y++)
                {
                    var block = blocks[block_index] = new float[BlockSize];
                    if (labels != null) block_labels[block_index] = labels[row];
                    block_index++;

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

            // Find kernels.
            var Kernels = new AffineKernel[NumKernels];
            for (int i = 0; i < NumKernels; i++)
            {
                Console.WriteLine($"Searching for kernel {i+1} of {NumKernels}");

                var kernel = Kernels[i] = FindKernel(blocks, BlockSize, labels: block_labels, use_labels: true);
                //kernel.Plot();

                var remainder = SetRemainder(blocks, kernel, GroupThreshold, labels: block_labels);
                blocks = remainder;
            }

            return Kernels;
        }

        static float[][,,] LevelOutput(Kernel[] Kernels, float[][,,] input, int Width, int Height, int Channels,
            int NumKernels = 3, int BlockDim = 3)
        {
            int BlockSize = BlockDim * BlockDim * Channels;

            // Construct level 1 output
            var output = new float[Rows][,,];
            var output_dim = Width - (BlockDim - 1);
            var block = new float[BlockSize];

            for (int row = 0; row < Rows; row++)
            {
                var image = input[row];
                var _image = new float[output_dim, output_dim, NumKernels];
                output[row] = _image;

                for (int x = 0; x < Width - (BlockDim - 1); x++)
                for (int y = 0; y < Height - (BlockDim - 1); y++)
                {
                    int i = 0;
                    for (int _x = x; _x < x + BlockDim; _x++)
                    for (int _y = y; _y < y + BlockDim; _y++)
                    for (int _c = 0; _c < Channels; _c++)
                    {
                        block[i++] = (float)image[_x, _y, _c];
                    }

                    for (int k = 0; k < NumKernels; k++)
                    {
                        var kernel = Kernels[k];
                        float score = kernel.Score(block);

                        if (score < ApplyThreshold)
                        {
                            _image[x, y, k] = kernel.Apply(block);
                            block = kernel.Remainder(block);
                        }
                        else
                        {
                            _image[x, y, k] = 0;
                        }
                    }
                }
            }

            return output;
        }

        static float TestKernel(float[][] blocks, Kernel kernel,
            float threshold, int step = 1)
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

        static Tuple<float, float> TestKernelWithLabels(float[][] blocks, Kernel kernel, byte[] labels,
            float threshold, int step = 1)
        {
            int
                in_count = 0, in_total = 0,
                out_count = 0, out_total = 0;

            for (int i = rnd.Next(step); i < blocks.Length; i += step)
            {
                var block = blocks[i];
                var label = labels[i];

                float error = kernel.Score(block);

                if (label == 0)
                {
                    in_total++;

                    if (error < threshold) in_count++;
                }
                else
                {
                    out_total++;

                    if (error < threshold) out_count++;
                }
            }

            float in_ratio = in_count / (float)in_total;
            float out_ratio = out_count / (float)out_total;

            return new Tuple<float, float>(in_ratio, in_ratio / out_ratio);
        }

        static float[][] SetRemainder(float[][] blocks, Kernel kernel, float threshold, byte[] labels = null)
        {
            bool[] in_remainder = new bool[blocks.Length];
            int[] label_count = new int[NumLabels];

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
                    if (labels != null) label_count[labels[i]]++;
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

            if (labels != null)
            {
                string s = "Label distribution among well fit: ";
                for (int i = 0; i < NumLabels; i++) s += (float)label_count[i] / (float)(blocks.Length - remainder.Length) + ", ";
                Console.WriteLine(s + '\n');
            }

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

        static AffineKernel FindKernel(float[][] blocks, int BlockSize, byte[] labels = null, bool use_labels = false)
        {
            float best_score = 0;
            AffineKernel best_kernel = null;
            for (int i = 0; i < 5000; i++)
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

                if (use_labels)
                {
                    var score = TestKernelWithLabels(blocks, kernel, labels, GroupThreshold,
                        //step: 599);
                        step: 1);

                    if (score.Item1 > .05f && score.Item2 > best_score)
                    {
                        best_score = score.Item2;
                        best_kernel = kernel;

                        Console.WriteLine($"(Iteration {i}) New best with a score of {best_score}.");
                    }
                }
                else
                {
                    var score = TestKernel(blocks, kernel,
                        GroupThreshold, step: 599);

                    if (score > best_score)
                    {
                        best_score = score;
                        best_kernel = kernel;

                        Console.WriteLine($"(Iteration {i}) New best with a score of {best_score}.");
                    }
                }
            }

            return best_kernel;
        }
    }
}
