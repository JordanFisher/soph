using System;
using System.IO;
using System.Diagnostics;
using System.Collections.Generic;
using System.Linq;

using System.Threading;
using System.Threading.Tasks;

using static Sophonautical.Util;

namespace Sophonautical
{
    public class SupervisedHypothesisGeneration
    {
        public static void SupervisedKernelHg(int Rows, LabeledImage[] images)
        {
            // Select training set and test set.
            const int TrainingSize = 9000;
            int TestSize = Rows - TrainingSize;
            var TrainingSet = Subset(Rows, images, TrainingSize);
            var TestSet = Remainder(images, TrainingSet);

            // Select smaller set to generate leads with.
            const int LeadSize = 700;
            const int InLabelSize = LeadSize / 2;

            var results = Enumerable.Range(0, NumLabels).AsParallel().Select(
                label => LearnKernelListForLabel(label, TrainingSize, TrainingSet, LeadSize, InLabelSize)
            ).ToList();

            // Sanity check.
            //var results = new List<Tuple<int, List<Tuple<AffineKernel, float>>>>();
            //for (int label = 0; label < NumLabels; label++)
            //{
            //    results.Add(new Tuple<int, List<Tuple<AffineKernel, float>>>(label, new List<Tuple<AffineKernel, float>>()));
            //}
            //results[0] = LearnKernelListForLabel(0, TrainingSize, TrainingSet, LeadSize, InLabelSize);

            // Sort the results by their label and unpack the tuple so we have
            // just an array of kernel arrays, indexed by label.
            results.Sort((list1, list2) => list1.Item1.CompareTo(list2.Item1));
            var kernel_lists = results.Select(klist => klist.Item2).ToList();

            // Truncate kernel lists so they're all the same length.
            const int desired_num_kernels = 20;
            foreach (var kernel_list in kernel_lists)
            {
                if (kernel_list.Count > desired_num_kernels)
                {
                    kernel_list.RemoveRange(desired_num_kernels, kernel_list.Count - desired_num_kernels);
                }
                else
                {
                    Console.WriteLine($"Warning! Kernel list only has {kernel_list.Count} kernels, fewer than {desired_num_kernels}");
                }
            }

            // Get test blocks.
            const int BlockDim = 5;
            var blocked_images = GetBlocks(TestSet, TestSize, Width, Height, Channels: 3, BlockDim: BlockDim, AddMirrors: true);

            // Test the kernel array for a label.
            for (int label = 0; label < NumLabels; label++)
            {
                Console.WriteLine($"Inspecting kernel list for label {label}");
                var list = kernel_lists[label];

                foreach (var kernelScore in list)
                {
                    Console.WriteLine($"{kernelScore.Item2}");
                }

                foreach (var kernel in list)
                {
                    var result = GetScoreStats(TestSize, blocked_images, kernel.Item1, label);
                    Console.WriteLine($"Score is {result.InOutHitRatio} with an in ratio of {result.InHitRatio}");
                }
            }

            // Test candidates on test set.
            int Right = 0, Wrong = 0;
            for (int i = 0; i < TestSize; i++)
            {
                var image = blocked_images[i];
                int guess = ClassifyImage(image, kernel_lists);

                if (guess == image.Label)
                {
                    Right++;
                }
                else
                {
                    Wrong++;
                }
            }
            float accuracy = (float)Right / (Right + Wrong);
            Console.WriteLine($"{Right} right vs {Wrong} wrong, {accuracy*100}%");
        }

        static Tuple<int, List<Tuple<AffineKernel, float>>> LearnKernelListForLabel(int label, int TrainingSize, LabeledImage[] TrainingSet, int LeadSize, int InLabelSize)
        {
            LabeledImage[] LeadSet = SampleSet(TrainingSize, TrainingSet, label, LeadSize, InLabelSize);

            // Get list of candidate kernels from the lead set.
            var kernel_list = Learn_SupervisedKernelHg(LeadSize, LeadSet, InLabel: label, InLabelMinRatio: 0.2f, SearchLength: 2000);
            kernel_list.Sort((kernelScore1, kernelScore2) => kernelScore2.Item2.CompareTo(kernelScore1.Item2));

            return new Tuple<int, List<Tuple<AffineKernel, float>>>(label, kernel_list);
        }

        public static void SupervisedKernelHgForLabel(int Rows, LabeledImage[] images, int InLabel)
        {
            // Select training set and test set.
            const int TrainingSize = 9000;
            int TestSize = Rows - TrainingSize;
            var TrainingSet = Subset(Rows, images, TrainingSize);
            var TestSet = Remainder(images, TrainingSet);

            // Select smaller set to generate leads with.
            const int LeadSize = 200;
            const int InLabelSize = LeadSize / 2;

            LabeledImage[] LeadSet = SampleSet(TrainingSize, TrainingSet, InLabel, LeadSize, InLabelSize);

            // Get list of candidate kernels from the lead set.
            var list = Learn_SupervisedKernelHg(LeadSize, LeadSet, InLabel: InLabel, InLabelMinRatio: 0.2f, SearchLength: 50);
            list.Sort((kernelScore1, kernelScore2) => kernelScore2.Item2.CompareTo(kernelScore1.Item2));

            foreach (var kernelScore in list)
            {
                Console.WriteLine($"{kernelScore.Item2}");
            }

            // Test candidates on test set.
            const int BlockDim = 5;
            var blocks = GetBlocks(TestSet, TestSize, Width, Height, Channels: 3, BlockDim: BlockDim, AddMirrors: true);

            foreach (var kernel in list)
            {
                var tracker = new BlockTracker();
                var result = GetScoreStats(TestSize, blocks, kernel.Item1, InLabel, tracker);
                Console.WriteLine($"Score is {result.InOutHitRatio} with an in ratio of {result.InHitRatio}");

                //pl.plot(Average(tracker.in_remainders), Average(tracker.out_remainders));
                //pl.plot(Average(tracker.in_blocks), Average(tracker.out_blocks));
                pl.plot(minus(Average(tracker.in_remainders), Average(tracker.out_remainders)));
                pl.plot(minus(Average(tracker.in_blocks), Average(tracker.out_blocks)));

                //var score = OptimizeKernelCutoff(TestSize, blocks, kernel.Item1, InLabel: InLabel, InLabelMinRatio: 0.2f);
                //Console.WriteLine($"Score is {score.InOutHitRatio} with an in ratio of {score.InHitRatio}");
            }
        }

        static List<Tuple<AffineKernel, float>> Learn_SupervisedKernelHg(int Rows, LabeledImage[] images, int InLabel,
            float InLabelMinRatio = 0.05f, int SearchLength = 1000000, int preferred_source = -1)
        {
            const int BlockDim = 5;
            const int Channels = 3;
            const int BlockSize = BlockDim * BlockDim * Channels;
            var blocked_images = GetBlocks(images, Rows, Width, Height, Channels: 3, BlockDim: BlockDim, AddMirrors: true);

            var list = new List<Tuple<AffineKernel, float>>();

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

                var score = OptimizeKernelCutoff(Rows, blocked_images, kernel,
                    InLabel: InLabel, InLabelMinRatio: InLabelMinRatio);

                if (score.InHitRatio > InLabelMinRatio)
                {
                    list.Add(new Tuple<AffineKernel, float>(kernel, score.InOutHitRatio));

                    if (score.InOutHitRatio > best_score)
                    {
                        best_score = score.InOutHitRatio;
                        best_kernel = kernel;

                        Console.WriteLine($"(Iteration {i}) New best with a score of {best_score}, in label ratio of {score.InHitRatio} with a threshold of {kernel.ClassificationCutoff}.");
                        best_kernel.Plot();
                    }
                }
            }

            return list;
        }

        static int ClassifyImage(BlockedImage image, List<List<Tuple<AffineKernel, float>>> kernel_lists)
        {
            var blocks = image.Blocks;
            float[] hits = new float[NumLabels];

            for (int label = 0; label < NumLabels; label++)
            {
                var kernel_list = kernel_lists[label];

                foreach (var kernel in kernel_list)
                {
                    float score = GetScore(image, kernel.Item1);
                    if (score < kernel.Item1.ClassificationCutoff)
                    {
                        //hits[i]++;
                        hits[label] += (float)Math.Log(kernel.Item2);
                    }
                }
            }

            int argmax = -1; float max = 0;
            for (int label = 0; label < NumLabels; label++)
            {
                if (argmax < 0 || hits[label] > max)
                {
                    argmax = label;
                    max = hits[label];
                }
            }

            Console.WriteLine($"Hits {s(hits)}");
            Console.WriteLine($"  Classify as {argmax}, actually a {image.Label}");

            return argmax;
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

                kernel.w[j] = w_block[j] - b_block[j];
                kernel.b[j] = b_block[j];

                //kernel.w[j] = w_block[j];
                //kernel.b[j] = 0;

                //kernel.w[j] = 1;
                //kernel.b[j] = b_block[j];
            }

            Normalize(kernel.w);

            return kernel;
        }

        static ScoreStats OptimizeKernelCutoff(int Rows, BlockedImage[] blocks, Kernel kernel, int InLabel,
            float InLabelMinRatio = 0.05f)
        {
            float[] source_score = GetScores(Rows, blocks, kernel);
            var labels = get_labels(blocks);

            ScoreStats best = null;

            for (int i = 0; i < 1000; i++)
            {
                float threshold = .001f + .5f * (float)rnd.NextDouble();
                var result = TestThreshold(Rows, kernel, source_score, labels, threshold, InLabel);

                if (best == null || result.InHitRatio > InLabelMinRatio && result.InOutHitRatio > best.InOutHitRatio)
                {
                    best = result;
                    kernel.ClassificationCutoff = threshold;
                }
            }

            return best;
        }

        static float GetScore(BlockedImage image, Kernel kernel, PostScoreAction action = null)
        {
            var blocks = image.Blocks;
            float score = float.MaxValue;

            for (int i = 0; i < blocks.Length; i++)
            {
                var block = blocks[i];

                float error = kernel.Score(block, action);
                score = Math.Min(score, error);
            }

            return score;
        }

        static float[] GetScores(int Rows, BlockedImage[] images, Kernel kernel)
        {
            float[] scores = new float[Rows];

            for (int source = 0; source < images.Length; source++)
            {
                scores[source] = GetScore(images[source], kernel);
            }

            return scores;
        }

        static ScoreStats GetScoreStats(int Rows, BlockedImage[] images, Kernel kernel, int InLabel, BlockTracker tracker = null)
        {
            var stats = new ScoreStats();

            for (int i = 0; i < Rows; i++)
            {
                var image = images[i];

                var action = tracker == null ? null : tracker.GetTrackInOut(image, InLabel);
                float error = GetScore(image, kernel, action: action);

                if (image.Label == InLabel)
                {
                    stats.InCount++;

                    if (error < kernel.ClassificationCutoff) stats.InHit++;
                }
                else
                {
                    stats.OutCount++;

                    if (error < kernel.ClassificationCutoff) stats.OutHit++;
                }
            }

            Console.WriteLine($"  - in label {stats.InHit} / {stats.InCount}  - out label {stats.OutHit} / {stats.OutCount}  - {kernel.ShortDescription()}");

            return stats;
        }

        static ScoreStats TestThreshold(int Rows, Kernel kernel, float[] source_score, byte[] source_label, float threshold, int InLabel)
        {
            var stats = new ScoreStats();

            for (int i = 0; i < Rows; i++)
            {
                var label = source_label[i];
                var error = source_score[i];

                if (label == InLabel)
                {
                    stats.InCount++;

                    if (error < threshold) stats.InHit++;
                }
                else
                {
                    stats.OutCount++;

                    if (error < threshold) stats.OutHit++;
                }
            }

            //Console.WriteLine($"  - in label {stats.InHit} / {stats.InCount}  - out label {stats.OutHit} / {stats.OutCount}  - error threshold {threshold:0.0000}  - {kernel.ShortDescription()}");

            return stats;
        }
    }
}
