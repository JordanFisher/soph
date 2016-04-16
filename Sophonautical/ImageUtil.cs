using System.Diagnostics;
using System.Collections.Generic;
using System.Linq;

namespace Sophonautical
{
    public partial class Util
    {
        public static T[] Shuffle<T>(T[] array)
        {
            return array.OrderBy(element => rnd.Next()).ToArray();
        }

        public static Dictionary<byte, List<LabeledImage>> OrganizeByLabel(int Rows, LabeledImage[] Images)
        {
            var ImagesByLabel = new Dictionary<byte, List<LabeledImage>>();

            for (byte l = 0; l < NumLabels; l++)
            {
                ImagesByLabel.Add(l, new List<LabeledImage>());
            }

            for (int i = 0; i < Rows; i++)
            {
                ImagesByLabel[Images[i].Label].Add(Images[i]);
            }

            return ImagesByLabel;
        }

        public static LabeledImage[] Subset(int Rows, LabeledImage[] Images, int SubsetSize)
        {
            var ImagesByLabel = OrganizeByLabel(Rows, Images);
            
            var LeadSet = new LabeledImage[SubsetSize];

            int count = 0;
            byte label = 0;
            while (count < SubsetSize)
            {
                var image = ImagesByLabel[label].Last();
                LeadSet[count] = image;
                ImagesByLabel[label].Remove(image);

                count++; label++;
                if (label >= NumLabels) label = 0;
            }

            Debug.Assert(ImagesByLabel.Sum(pair => pair.Value.Count) == Rows - SubsetSize);

            return Shuffle(LeadSet);
        }

        public static LabeledImage[] Remainder(LabeledImage[] FullSet, LabeledImage[] OtherSet)
        {
            LabeledImage[] RemainderSet = new LabeledImage[FullSet.Length - OtherSet.Length];

            int index = 0;
            foreach (var image in FullSet)
            {
                if (!OtherSet.Contains(image))
                {
                    RemainderSet[index++] = image;
                }
            }

            return Shuffle(RemainderSet);
        }

        public static LabeledImage[] SampleSet(int Rows, LabeledImage[] Images, int InLabel, int SampleSize, int InLabelSize)
        {
            var LeadSet = new LabeledImage[SampleSize];

            int lead_index = 0, source_index = rnd.Next(0, Rows);
            while (lead_index < SampleSize)
            {
                source_index++; if (source_index >= Rows) source_index = 0;

                if (lead_index < InLabelSize && Images[source_index].Label == InLabel ||
                    lead_index >= InLabelSize && Images[source_index].Label != InLabel)
                {
                    LeadSet[lead_index] = Images[source_index];
                    lead_index++;
                }
            }

            return LeadSet;
        }
    }
}
