﻿using System;
using System.IO;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Collections.Generic;

using static Sophonautical.Util;

namespace Sophonautical
{
    public abstract class Kernel
    {
        public float ClassificationCutoff = 1;

        public virtual float Score(float[] input, PostScoreAction action = null)
        {
            var remainder = Remainder(input);

            //var error = NormInf(remainder);
            //var error = Norm1(remainder) / input.Length;
            var error = Norm1Thresh(remainder) / input.Length;

            if (action != null) action(this, input, error, remainder);

            return error;
        }

        public virtual float[] Reconstruct(float[] input)
        {
            return Inverse(Apply(input));
        }

        public virtual float[] Remainder(float[] input)
        {
            //return FuzzyDiff(input, Reconstruct(input), channels: 3);
            return minus(input, Reconstruct(input));
        }

        public abstract float ArgMinInverseError(float[] input);
        public abstract float Apply(float[] input);
        public abstract float[] Inverse(float y);
        public abstract void Plot();

        public abstract string ShortDescription();
    }

    public class AffineKernel : Kernel
    {
        public float[] w, b;
        public float threshold = 0.5f;

        public AffineKernel(float[] w, float[] b = null) { this.w = w; this.b = b; }

        public AffineKernel(int BlockSize)
        {
            w = new float[BlockSize];
            b = new float[BlockSize];
        }

        public override float Score(float[] input, PostScoreAction action = null)
        {
            var remainder = Remainder(input);
            var error = Norm1Thresh(remainder, threshold: threshold) / input.Length;

            if (action != null) action(this, input, error, remainder);

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
            return $"l1thresh {threshold:0.0000}  class-threh {ClassificationCutoff:0.0000}";
        }
    }
}