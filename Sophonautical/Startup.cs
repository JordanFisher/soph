using System;
using System.IO;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;

namespace Sophonautical
{
    public partial class Util
    {
        public const string
            OutputDir = "C://Users//Jordan//Desktop//output",
            TrainingDataDir = "C://Users//Jordan//Desktop//cifar-10-batches-bin",
            LabelFile = "batches.meta.txt",
            FileName = "data_batch_1.bin";

        public const int
            Width = 32, Height = 32,
            Rows = 10000,
            RowSize = 3073,
            NumLabels = 10;

        public static void Startup()
        {
            pl = new Plot(OutputDir, Quiet: false, Save: true);
            //pl = new Plot(OutputDir, Quiet: true, Save: true);
            rnd = new Random(1);
        }

        public static LabeledImage[] Init()
        {
            // Read label meta file.
            var LabelNames = File.ReadAllLines(Path.Combine(TrainingDataDir, LabelFile));

            // Read training data.
            var bytes = File.ReadAllBytes(Path.Combine(TrainingDataDir, FileName));
            Debug.Assert(bytes.Length == Rows * RowSize);

            // Unpack training data.
            LabeledImage[] images = new LabeledImage[Rows];
            var source_index = 0;

            for (int row = 0; row < Rows; row++)
            {
                var image = images[row] = new LabeledImage();
                var pixels = image.Pixels = new float[Width, Height, 3];

                image.Label = bytes[source_index++];

                for (int channel = 0; channel < 3; channel++)
                for (int x = 0; x < Width; x++)
                for (int y = 0; y < Height; y++)
                {
                    pixels[x, y, channel] = bytes[source_index++] / 255f;
                }

                //// Swap out source images for derivatives.
                //var diff = new float[Width, Height, 3];
                //for (int x = 0; x < Width; x++)
                //for (int y = 0; y < Height; y++)
                //{
                //    diff[x, y, 0] = .33333f * (pixels[x, y, 0] + pixels[x, y, 0] + pixels[x, y, 0]);
                //    diff[x, y, 1] = 0;
                //    diff[x, y, 2] = 0;
                //}
                //for (int x = 1; x < Width - 1; x++)
                //for (int y = 1; y < Height - 1; y++)
                //{
                //    diff[x, y, 1] = 10 * (pixels[x + 1, y, 0] - pixels[x, y, 0]);
                //    diff[x, y, 2] = 10 * (pixels[x, y + 1, 0] - pixels[x, y, 0]);
                //}
                //for (int x = 0; x < Width; x++)
                //for (int y = 0; y < Height; y++)
                //{
                //    diff[x, y, 0] = 0;
                //}

                //images[row].Pixels = diff;
            }

            Debug.Assert(source_index == Rows * RowSize);

            return images;
        }

        public static BlockedImage[] GetBlocks(LabeledImage[] inputs, int Rows, int Width, int Height, int Channels,
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
    }
}