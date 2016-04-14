using System;
using System.IO;
using System.Diagnostics;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

using static Sophonautical.Util;

namespace Sophonautical
{
    public class SupervisedHypothesisGeneration
    {
        public static void Learn_SupervisedKernelHG_Staged(int Rows, LabledImage[] images,
            int InLabel = 0)
        {
            // Select smaller set to generate leads with.
            const int LeadSize = 200;
            const int InLabelSize = LeadSize / 2;
            const int OutLabelSize = LeadSize - InLabelSize;

            LabledImage[] LeadSet = new LabledImage[LeadSize];

            int RemainderSize = Rows - LeadSize;
            float[][,,] RemainderSet = new float[RemainderSize][,,];
            byte[] RemainderLabels = new byte[RemainderSize];

            int lead_index = 0, source_index = rnd.Next(0, Rows);
            while (lead_index < LeadSize)
            {
                source_index++; if (source_index >= Rows) source_index = 0;

                if (lead_index < InLabelSize && images[source_index].Label == InLabel ||
                    lead_index >= InLabelSize && images[source_index].Label != InLabel)
                {
                    LeadSet[lead_index] = images[source_index];
                    lead_index++;
                }
            }

            // Get list of candidate kernels from the lead set.
            var list = Learn_SupervisedKernelHG(LeadSize, LeadSet, InLabelMinRatio: 0.2f, SearchLength: 1000);
            list.Sort((kernelScore1, kernelScore2) => kernelScore2.Item2.CompareTo(kernelScore1.Item2));

            foreach (var kernelScore in list)
            {
                Console.WriteLine($"{kernelScore.Item2}");
            }

            // Test candidates on bigger test set.
            const int BlockDim = 5;
            var blocks = GetBlocks(images, Rows / 10, Width, Height, Channels: 3, BlockDim: BlockDim, AddMirrors: true);

            foreach (var kernel in list)
            {
                //Console.WriteLine("!!!");
                //int w_index = 0; do w_index = rnd.Next(0, blocks.Length - 1); while (labels[w_index] != InLabel);
                //var w_block = blocks[w_index][rnd.Next(0, blocks[w_index].Length)];
                //var _kernel = (AffineKernel)FormHypothesis(blocks[0][0].Length, w_block, list[1].Item1.b);
                //var score = TestKernelWithSources(Rows / 10, blocks, _kernel, labels,
                //    InLabel: InLabel, step: 1, InLabelMinRatio: 0.05f, verbose: true);

                var score = TestKernelWithSources(Rows / 10, blocks, kernel.Item1,
                    InLabel: InLabel, step: 1, InLabelMinRatio: 0.05f, verbose: true);
                Console.WriteLine($"Score is {score}");
            }
        }

        static List<Tuple<AffineKernel, float>> Learn_SupervisedKernelHG(int Rows, LabledImage[] images,
            int InLabel = 0, float InLabelMinRatio = 0.05f, int SearchLength = 1000000, int preferred_source = -1)
        {
            const int BlockDim = 5;
            const int Channels = 3;
            const int BlockSize = BlockDim * BlockDim * Channels;
            var blocked_images = GetBlocks(images, Rows, Width, Height, Channels: 3, BlockDim: BlockDim, AddMirrors: true);

            List<Tuple<AffineKernel, float>> list = new List<Tuple<AffineKernel, float>>();

            Console.WriteLine("\n\nSearching for kernels via supervised hypothesis generation\n\n");

            float best_score = 0;
            AffineKernel best_kernel = null;

            for (int i = 0; i < SearchLength; i++)
            {
                int w_index, b_index;
                if (preferred_source >= 0)
                {
                    w_index = b_index = preferred_source;
                }
                else
                {
                    do w_index = rnd.Next(0, blocked_images.Length - 1); while (images[w_index].Label != InLabel);
                    do b_index = rnd.Next(0, blocked_images.Length - 1); while (images[b_index].Label != InLabel);
                }

                var w_blocks = blocked_images[w_index].Blocks;
                var b_blocks = blocked_images[b_index].Blocks;

                var w_block = w_blocks[rnd.Next(0, w_blocks.Length)];
                var b_block = b_blocks[rnd.Next(0, b_blocks.Length)];

                var kernel = (AffineKernel)FormHypothesis(BlockSize, w_block, b_block);

                var score = TestKernelWithSources(Rows, blocked_images, kernel,
                    InLabel: InLabel, step: 1, InLabelMinRatio: InLabelMinRatio);

                if (score.Item1 > InLabelMinRatio)
                {
                    list.Add(new Tuple<AffineKernel, float>(kernel, score.Item2));

                    if (score.Item2 > best_score)
                    {
                        best_score = score.Item2;
                        best_kernel = kernel;

                        Console.WriteLine($"(Iteration {i}) New best with a score of {best_score}, in label ratio of {score.Item1} with a threshold of {score.Item3}.");
                        best_kernel.Plot();
                    }
                }
            }

            return list;
        }

        static Kernel FormHypothesis(int BlockSize, float[] w_block, float[] b_block)
        {
            var kernel = new AffineKernel(BlockSize);

            for (int j = 0; j < BlockSize; j++)
            {
                kernel.threshold = .01f + .5f * (float)rnd.NextDouble();
                //kernel.threshold = 1;

                //kernel.w[j] = (float)rnd.NextDouble();
                //kernel.b[j] = (float)rnd.NextDouble();

                //kernel.w[j] = w_block[j] - b_block[j];
                //kernel.b[j] = b_block[j];

                //kernel.w[j] = w_block[j];
                //kernel.b[j] = 0;

                kernel.w[j] = 1;
                kernel.b[j] = b_block[j];
            }

            Normalize(kernel.w);

            return kernel;
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

        static Tuple<float, float, float> TestKernelWithSources(int Rows, BlockedImage[] blocks, Kernel kernel, int InLabel,
            float InLabelMinRatio = 0.05f, int step = 1, int offset = 0, bool verbose = false)
        {
            float[] source_score = new float[Rows];
            float default_error = 9999999;
            for (int i = offset; i < offset + Rows; i++) source_score[i] = default_error;

            for (int source = rnd.Next(step); source < blocks.Length; source += step)
            {
                var _blocks = blocks[source].Blocks;

                for (int j = 0; j < _blocks.Length; j++)
                {
                    var block = _blocks[j];

                    float error = kernel.Score(block);
                    source_score[source] = Math.Min(source_score[source], error);
                }
            }

            float best_threshold = -1, best_ratio = -1, best_in_ratio = -1;
            var labels = get_labels(blocks);
            for (int i = 0; i < 1000; i++)
            {
                float threshold = .01f + .14f * (float)rnd.NextDouble();
                var result = TestThreshold(Rows, kernel, source_score, labels, threshold, InLabel, offset: offset);

                if (result.Item1 > InLabelMinRatio && result.Item2 > best_ratio || best_ratio < 0)
                {
                    best_in_ratio = result.Item1;
                    best_ratio = result.Item2;
                    best_threshold = threshold;
                }
            }
            TestThreshold(Rows, kernel, source_score, labels, best_threshold, InLabel, verbose: true, offset: offset);

            if (verbose)
            {
                // Stat breakdown.
                Console.WriteLine("(");
                TestThreshold(Rows, kernel, source_score, labels, best_threshold, InLabel, verbose: true, offset: 0, step: 2);
                TestThreshold(Rows - 1, kernel, source_score, labels, best_threshold, InLabel, verbose: true, offset: 1, step: 2);
                Console.WriteLine(")");
            }

            return new Tuple<float, float, float>(best_in_ratio, best_ratio, best_threshold);
        }

        struct LabelRatioPerformance
        {
            float InRatio, Ratio, Threshold;
        }

        private static Tuple<float, float> TestThreshold(int Rows, Kernel kernel, float[] source_score, byte[] source_label, float threshold, int InLabel,
            bool verbose = false, int offset = 0, int step = 1)
        {
            int in_count = 0, in_total = 0, out_count = 0, out_total = 0;

            for (int i = offset; i < offset + Rows; i += step)
            {
                var label = source_label[i];
                var error = source_score[i];

                if (label == InLabel)
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

            float in_ratio = in_count / ((float)in_total + .001f);
            float out_ratio = out_count / ((float)out_total + .001f);

            if (verbose)
            {
                Console.WriteLine($"  - in label {in_count} / {in_total}  - out label {out_count} / {out_total}  - error threshold {threshold:0.0000}  - {kernel.ShortDescription()}");
            }

            return new Tuple<float, float>(in_ratio, in_ratio / out_ratio);
        }
    }
}
