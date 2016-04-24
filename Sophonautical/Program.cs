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
        static void Test()
        {
            return;

            UtilTest.Run();

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
            Startup(); Test();

            LabeledImage[] images = Init();

            //SupervisedHypothesisGeneration.SupervisedKernelHg(Rows, images);
            SupervisedHypothesisGeneration.SupervisedKernelHgForLabel(Rows, images, InLabel: 7);

            //Learn_Multilevel(images, labels);

            Console.ReadLine();
        }
    }
}
