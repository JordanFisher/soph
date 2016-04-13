using System;
using System.IO;
using System.Diagnostics;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

using static Sophonautical.Util;

namespace Sophonautical
{
    public partial class Program
    {
        static void Learn_Multilevel(float[][,,] images)
        {
            Console.WriteLine("\n\nLearning level 1\n\n");

            int block_dim1 = 5, num_kernels1 = 12;
            var output_dim1 = Width - (block_dim1 - 1);

            var kernels1 = LearnLevel(images, Width, Height, 3,
                NumKernels: num_kernels1, BlockDim: block_dim1);
            var output1 = LevelOutput(kernels1, images, Width, Height, 3,
                NumKernels: num_kernels1, BlockDim: block_dim1);
            NormalizeComponents(output1, Rows, output_dim1, output_dim1, num_kernels1);

            Console.WriteLine("\n\nLearning level 2\n\n");

            int block_dim2 = 3, num_kernels2 = 6;
            var output_dim2 = output_dim1 - (block_dim2 - 1);

            var kernels2 = LearnLevel(output1, output_dim1, output_dim1, num_kernels1,
                NumKernels: num_kernels2, BlockDim: block_dim2);
            var output2 = LevelOutput(kernels2, output1, output_dim1, output_dim1, num_kernels1,
                NumKernels: num_kernels2, BlockDim: block_dim2);
            NormalizeComponents(output2, Rows, output_dim2, output_dim2, num_kernels2);

            Console.WriteLine("\n\nLearning level 3\n\n");

            int block_dim3 = 3, num_kernels3 = 3;
            var output_dim3 = output_dim2 - (block_dim3 - 1);

            var kernels3 = LearnLevel(output2, output_dim2, output_dim2, num_kernels2,
                NumKernels: num_kernels3, BlockDim: block_dim3);
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
            int NumKernels = 3, int BlockDim = 3)
        {
            int BlockSize = BlockDim * BlockDim * Channels;

            // Unpack blocks.
            int num_blocks = Rows * (Width - (BlockDim - 1)) * (Height - (BlockDim - 1));
            var blocks = new float[num_blocks][];
            var block_labels = new byte[num_blocks];
            var block_sources = new int[num_blocks];
            int block_index = 0;

            for (int row = 0; row < Rows; row++)
            {
                var image = input[row];

                for (int x = 0; x < Width - (BlockDim - 1); x++)
                    for (int y = 0; y < Height - (BlockDim - 1); y++)
                    {
                        var block = blocks[block_index] = new float[BlockSize];

                        block_sources[block_index] = row;
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
                Console.WriteLine($"Searching for kernel {i + 1} of {NumKernels}");

                var kernel = Kernels[i] = FindKernel(blocks, BlockSize);
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

        static AffineKernel FindKernel(float[][] blocks, int BlockSize, int[] sources = null)
        {
            float best_score = 0;
            AffineKernel best_kernel = null;
            for (int i = 0; i < 500000; i++)
            {
                var kernel = new AffineKernel(BlockSize);

                int w_index, b_index;
                w_index = rnd.Next(0, blocks.Length - 1);
                b_index = rnd.Next(0, blocks.Length - 1);
                for (int j = 0; j < BlockSize; j++)
                {
                    //kernel.w[j] = (float)rnd.NextDouble();
                    //kernel.b[j] = (float)rnd.NextDouble();

                    kernel.w[j] = blocks[w_index][j] - blocks[b_index][j];
                    kernel.b[j] = blocks[b_index][j];
                    kernel.threshold = .01f + .5f * (float)rnd.NextDouble();

                    //kernel.w[j] = blocks[w_index][j];
                    //kernel.b[j] = 0;
                }

                Normalize(kernel.w);

                var score = TestKernel(blocks, kernel,
                    GroupThreshold, step: 599);

                if (score > best_score)
                {
                    best_score = score;
                    best_kernel = kernel;

                    Console.WriteLine($"(Iteration {i}) New best with a score of {best_score}.");
                }
            }

            return best_kernel;
        }
    }
}
