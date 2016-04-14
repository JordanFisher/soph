using System;
using System.IO;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;

namespace Sophonautical
{
    public partial class Util
    {
        public class Image
        {
            public float[,,] Pixels;
        }

        public class LabeledImage
        {
            public float[,,] Pixels;
            public byte Label;
        }

        public class BlockedImage
        {
            public float[,,] Pixels;
            public byte Label;
            public float[][] Blocks;

            public BlockedImage(LabeledImage image)
            {
                this.Pixels = image.Pixels;
                this.Label = image.Label;
            }
        }

        public static float[][,,] get_pixels(LabeledImage[] images)
        {
            var pixels = new float[images.Length][,,];
            for (int i = 0; i < images.Length; i++)
            {
                pixels[i] = images[i].Pixels;
            }

            return pixels;
        }

        public static byte[] get_labels(BlockedImage[] images)
        {
            var labels = new byte[images.Length];
            for (int i = 0; i < images.Length; i++)
            {
                labels[i] = images[i].Label;
            }

            return labels;
        }
    }
}