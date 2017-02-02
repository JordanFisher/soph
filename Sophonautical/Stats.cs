using System;
using System.IO;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;

using static Sophonautical.Util;

namespace Sophonautical
{
    public class ScoreStats
    {
        public int InHit = 0, InCount = 0, OutHit = 0, OutCount = 0;

        public float Precision { get { return (float)InHit / (InHit + OutHit + .001f); } }
        public float Recall { get { return (float)InHit / (InCount + .001f); } }
        public float Accuracy { get { return (float)(InHit + OutReject) / (InCount + OutCount); } }

        public int InReject { get { return InCount - InHit; } }
        public int OutReject { get { return OutCount - OutHit; } }

        public float InHitRatio { get { return (float)InHit / (InCount + .001f); } }
        public float OutHitRatio { get { return (float)OutHit / (OutCount + .001f); } }
        public float InOutHitRatio { get { return InHitRatio / OutHitRatio; } }

        public override string ToString()
        {
            return $"Precision = {Precision} Recall = {Recall} Accuracy = {Accuracy} : In = {InHit}/{InCount} Out = {OutHit}/{OutCount}";
        }
    }

    public class ClassificationStats
    {
        public int Total = 0;
        int[,] Confusion = new int[NumLabels, NumLabels];

        public void Increment(int PredictedLabel, int ActualLabel)
        {
            Confusion[PredictedLabel, ActualLabel]++;
            Total++;
        }

        public int Hits(int PredictedLabel, int ActualLabel)
        {
            return Confusion[PredictedLabel, ActualLabel];
        }

        public ScoreStats StatsForClass(int label)
        {
            var stats = new ScoreStats();

            for (int l = 0; l <= NumLabels; l++)
            {
                stats.InCount += Hits(PredictedLabel: l, ActualLabel: label);
            }
            stats.OutCount = Total - stats.InCount;

            for (int l = 0; l <= NumLabels; l++)
            {
                if (l == label)
                {
                    stats.InHit += Hits(PredictedLabel: label, ActualLabel: l);
                }
                else
                {
                    stats.OutHit += Hits(PredictedLabel: label, ActualLabel: l);
                }
            }

            return stats;
        }

        public ScoreStats[] StatsForClasses()
        {
            var stats = new ScoreStats[NumLabels];
            for (int label = 0; label < NumLabels; label++)
            {
                stats[label] = StatsForClass(label);
            }

            return stats;
        }

        public string PredictionsForClassStr(int label)
        {
            string s = "";

            for (int l = 0; l < NumLabels; l++)
            {
                s += $"{Hits(PredictedLabel: l, ActualLabel: label):0000} ";
            }

            return s;
        }

        public string ConfusionMatrixStr()
        {
            string s = "";

            for (int l = 0; l < NumLabels; l++)
            {
                s += PredictionsForClassStr(l) + '\n';
            }

            return s;
        }
    }
}