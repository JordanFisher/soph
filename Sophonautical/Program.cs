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
        const string
            OutputDir = "C://Users//Jordan//Desktop//output",
            TrainingDataDir = "C://Users//Jordan//Desktop//cifar-10-batches-bin",
            LabelFile = "batches.meta.txt",
            FileName = "data_batch_1.bin";

        //static Plot pl = new Plot(OutputDir, Quiet: false, Save: false);
        static Plot pl = new Plot(OutputDir, Quiet: true, Save: true);
        static Random rnd = new Random(1);

        const int
            Width = 32, Height = 32,
            Rows = 10000,
            RowSize = 3073,
            NumLabels = 10;

        const float
            GroupThreshold = .075f,
            ApplyThreshold = 10 * GroupThreshold;

        class Image
        {
            public float[,,] Pixels;
        }

        class LabledImage
        {
            public float[,,] Pixels;
            public byte Label;
        }

        class BlockedImage
        {
            public float[,,] Pixels;
            public byte Label;
            public float[][] Blocks;

            public BlockedImage(LabledImage image)
            {
                this.Pixels = image.Pixels;
                this.Label = image.Label;
            }
        }

        static float[][,,] get_pixels(LabledImage[] images)
        {
            var pixels = new float[images.Length][,,];
            for (int i = 0; i < images.Length; i++)
            {
                pixels[i] = images[i].Pixels;
            }

            return pixels;
        }

        static byte[] get_labels(BlockedImage[] images)
        {
            var labels = new byte[images.Length];
            for (int i = 0; i < images.Length; i++)
            {
                labels[i] = images[i].Label;
            }

            return labels;
        }

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

            LabledImage[] images = Init();

            //Parallel.For(0, NumKernels, i =>
            Learn_SupervisedKernelHG_Staged(Rows, images, InLabel: 0);
            //Learn_SupervisedKernelHG(Rows, images, labels);
            //Learn_SupervisedKernelHG(5000, images, labels);
            //Learn_SupervisedKernelHG(200, images, labels, InLabelMinRatio: 0.3f);
            //Learn_Multilevel(images, labels);

            Console.ReadLine();
        }

        static LabledImage[] Init()
        {
            // Read label meta file.
            var LabelNames = File.ReadAllLines(Path.Combine(TrainingDataDir, LabelFile));

            // Read training data.
            var bytes = File.ReadAllBytes(Path.Combine(TrainingDataDir, FileName));
            Debug.Assert(bytes.Length == Rows * RowSize);

            // Unpack training data.
            LabledImage[] images = new LabledImage[Rows];
            var source_index = 0;

            for (int row = 0; row < Rows; row++)
            {
                var image = images[row] = new LabledImage();
                var pixels = image.Pixels = new float[Width, Height, 3];

                image.Label = bytes[source_index++];

                for (int channel = 0; channel < 3; channel++)
                for (int x = 0; x < Width; x++)
                for (int y = 0; y < Height; y++)
                {
                    pixels[x, y, channel] = bytes[source_index++] / 255f;
                }

                // Swap out source images for derivatives.
                var diff = new float[Width, Height, 3];
                for (int x = 0; x < Width; x++)
                for (int y = 0; y < Height; y++)
                {
                    diff[x, y, 0] = .33333f * (pixels[x, y, 0] + pixels[x, y, 0] + pixels[x, y, 0]);
                    diff[x, y, 1] = 0;
                    diff[x, y, 2] = 0;
                }
                for (int x = 1; x < Width - 1; x++)
                for (int y = 1; y < Height - 1; y++)
                {
                    diff[x, y, 1] = 10 * (pixels[x + 1, y, 0] - pixels[x, y, 0]);
                    diff[x, y, 2] = 10 * (pixels[x, y + 1, 0] - pixels[x, y, 0]);
                }
                for (int x = 0; x < Width; x++)
                for (int y = 0; y < Height; y++)
                {
                    diff[x, y, 0] = 0;
                }

                images[row].Pixels = diff;
            }

            Debug.Assert(source_index == Rows * RowSize);

            return images;
        }

        static BlockedImage[] GetBlocks(LabledImage[] inputs, int Rows, int Width, int Height, int Channels,
            int BlockDim, bool AddMirrors = false)
        {
            int BlockSize = BlockDim * BlockDim * Channels;

            // Unpack blocks.
            int num_blocks = (Width - (BlockDim - 1)) * (Height - (BlockDim - 1));
            if (AddMirrors) num_blocks *= 8;

            var blocked_images = new BlockedImage[Rows];

            for (int row = 0; row < Rows; row++)
            {
                var image = blocked_images[row] = new BlockedImage(inputs[row]);

                var pixels = image.Pixels;
                var image_blocks = image.Blocks = new float[num_blocks][];

                int block_index = 0;

                for (int mirror = 0; mirror < (AddMirrors ? 8 : 1); mirror++)
                for (int x = 0; x < Width - (BlockDim - 1); x++)
                for (int y = 0; y < Height - (BlockDim - 1); y++)
                {
                    var block = image_blocks[block_index] = new float[BlockSize];
                    block_index++;

                    int i = 0;
                    for (int _x = 0; _x < BlockDim; _x++)
                    for (int _y = 0; _y < BlockDim; _y++)
                    for (int _c = 0; _c < Channels; _c++)
                    {
                        if      (mirror == 0) block[i++] = (float)pixels[x + _x, y + _y, _c];
                        else if (mirror == 1) block[i++] = (float)pixels[x + BlockDim - 1 - _x, y + _y, _c];
                        else if (mirror == 2) block[i++] = (float)pixels[x + _x, y + BlockDim - 1 - _y, _c];
                        else if (mirror == 3) block[i++] = (float)pixels[x + BlockDim - 1 - _x, y + BlockDim - 1 - _y, _c];
                        else if (mirror == 4) block[i++] = (float)pixels[y + _y, x + _x, _c];
                        else if (mirror == 5) block[i++] = (float)pixels[y + BlockDim - 1 - _y, x + _x, _c];
                        else if (mirror == 6) block[i++] = (float)pixels[y + _y, x + BlockDim - 1 - _x, _c];
                        else if (mirror == 7) block[i++] = (float)pixels[y + BlockDim - 1 - _y, x + BlockDim - 1 - _x, _c];
                    }
                }

                Debug.Assert(block_index == blocked_images[row].Blocks.Length);
            }

            return blocked_images;
        }

        static void Learn_SupervisedKernelHG_Staged(int Rows, LabledImage[] images,
            int InLabel = 0)
        {
            // Select smaller set to generate leads with.
            const int LeadSize = 600;
            const int InLabelSize = LeadSize / 3;
            const int OutLabelSize = LeadSize - InLabelSize;

            LabledImage[] LeadSet = new LabledImage[LeadSize];

            int RemainderSize = Rows - LeadSize;
            float[][,,] RemainderSet = new float[RemainderSize][,,];
            byte[] RemainderLabels = new byte[RemainderSize];

            int lead_index = 0, source_index = rnd.Next(0, Rows);
            while (lead_index < LeadSize)
            {
                source_index++; if (source_index >= Rows) source_index = 0;

                if (lead_index <  InLabelSize && images[source_index].Label == InLabel ||
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

            public abstract string ShortDescription();
        }

        class AffineKernel : Kernel
        {
            public float[] w, b;
            public float threshold = GroupThreshold;

            public AffineKernel(float[] w, float[] b=null) { this.w = w; this.b = b; }

            public AffineKernel(int BlockSize)
            {
                w = new float[BlockSize];
                b = new float[BlockSize];
            }

            public override float Score(float[] input)
            {
                var remainder = Remainder(input);
                var error = Norm1Thresh(remainder, threshold: threshold) / input.Length;

                return error;
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

            public override string ShortDescription()
            {
                return $"l1thresh {threshold:0.0000}";
            }
        }
    }
}
