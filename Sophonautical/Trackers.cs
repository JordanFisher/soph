using System;
using System.IO;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Collections.Generic;

using static Sophonautical.Util;

namespace Sophonautical
{
    public delegate void PostScoreAction(Kernel kernel, float[] input = null, float error = -1, float[] remainder = null);
    public class BlockTracker
    {
        public List<float[]>
            in_blocks = new List<float[]>(),
            in_remainders = new List<float[]>(),
            out_blocks = new List<float[]>(),
            out_remainders = new List<float[]>();

        public PostScoreAction GetTrackInOut(BlockedImage image, int InLabel)
        {
            if (image.Label == InLabel)
            {
                return TrackRelevantInBlocks;
            }
            else
            {
                return TrackRelevantOutBlocks;
            }
        }

        public void TrackRelevantInBlocks(Kernel kernel, float[] input = null, float error = -1, float[] remainder = null)
        {
            if (error < kernel.ClassificationCutoff)
            {
                in_blocks.Add(input);
                in_remainders.Add(remainder);
            }
        }

        public void TrackRelevantOutBlocks(Kernel kernel, float[] input = null, float error = -1, float[] remainder = null)
        {
            if (error < kernel.ClassificationCutoff)
            {
                out_blocks.Add(input);
                out_remainders.Add(remainder);
            }
        }
    }
}