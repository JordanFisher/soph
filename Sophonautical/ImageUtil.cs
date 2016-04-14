using System;
using System.IO;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;

namespace Sophonautical
{
    public partial class Util
    {
        public static LabeledImage[] SampleSet(int Rows, LabeledImage[] images, int InLabel, int LeadSize, int InLabelSize)
        {
            var LeadSet = new LabeledImage[LeadSize];

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

            return LeadSet;
        }
    }
}
