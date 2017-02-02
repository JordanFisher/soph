using System;
using System.IO;
using System.Text;
using System.Diagnostics;
using System.Collections.Generic;
using System.Linq;

namespace Sophonautical
{
    public class Plot
    {
        string OutputDir;
        bool Quiet;
        bool Save;
        bool UsePlotNumber = false;

        int PlotCount = 0;

        public Plot(string OutputDir, bool Quiet = false, bool Save = false)
        {
            this.OutputDir = OutputDir;
            this.Quiet = Quiet;
            this.Save = Save;

            if (Quiet)
            {
                UsePlotNumber = true;
            }
        }

        void run_python(string file_name, params string[] args)
        {
            ProcessStartInfo start = new ProcessStartInfo();
            start.FileName = "python";
            start.Arguments = string.Format("{0} {1}", file_name, string.Join(" ", args));
            start.UseShellExecute = false;
            start.RedirectStandardOutput = true;
            start.WorkingDirectory = OutputDir;

            using (Process process = Process.Start(start))
            {
                using (StreamReader reader = process.StandardOutput)
                {
                    string result = reader.ReadToEnd();
                    Console.Write(result);
                }
            }
        }

        void run_plot(string py)
        {
            PlotCount++;

            string py_file;
            if (UsePlotNumber)
            {
                py_file = $"plot{PlotCount}.py";
            }
            else
            {
                py_file = $"plot.py";
            }

            string full_path = Path.Combine(OutputDir, py_file);

            File.WriteAllText(full_path, py);

            var args = new List<string>();
            if (Quiet) args.Add("quiet");
            if (Save) args.Add("save");

            run_python(py_file, args.ToArray());
        }

        public string py_array<T>(T[] data)
        {
            return '[' + string.Join(",", data) + ']';
        }

        public string py_array(float[,] data)
        {
            StringBuilder s = new StringBuilder();
            s.Append('[');

            for (int i = 0; i < data.GetLength(0); i++)
            {
                s.Append('[');

                for (int j = 0; j < data.GetLength(1); j++)
                {
                    s.Append(data[i, j]);
                    s.Append(',');
                }
                    
                s.Append(']');
                s.Append(',');
            }

            s.Append(']');

            return s.ToString();
        }

        public string py_array<T>(T[,,] data)
        {
            StringBuilder s = new StringBuilder();
            s.Append('[');

            for (int i = 0; i < data.GetLength(0); i++)
            {
                s.Append('[');

                for (int j = 0; j < data.GetLength(1); j++)
                {
                    s.Append('[');

                    s.Append(data[i, j, 0]);
                    s.Append(',');
                    s.Append(data[i, j, 1]);
                    s.Append(',');
                    s.Append(data[i, j, 2]);

                    s.Append(']');
                    s.Append(',');
                }

                s.Append(']');
                s.Append(',');
            }

            s.Append(']');

            return s.ToString();
        }

        string indent(string code)
        {
            code = code.Replace("\n\r", "\n").Replace("\r\n", "\n");

            var lines = code.Split('\n');

            var indented_code = "";
            foreach (var line in lines)
            {
                indented_code += '\t' + line + '\n';
            }

            return indented_code;
        }

        string py_plot(string plot)
        {
            string save_path = "hello.png";

            if (UsePlotNumber)
            {
                save_path = $"plot_{PlotCount}.png";
            }

            return string.Format($@"
import sys
import numpy as np
import matplotlib.pyplot as pl

def plot():
{indent(plot)}

if __name__ == '__main__':
    plot()

    if 'save' in sys.argv:
        pl.savefig('{save_path}')

    if not 'quiet' in sys.argv:
        pl.show()
"

            );
        }

        string line_plot(string data, string plot_ref = "pl")
        {
            return string.Format($@"
data = np.array({data})

{plot_ref}.plot(data)"
            );
        }

        string image_plot(string data, string plot_ref = "pl")
        {
            return string.Format($@"
data = np.array({data})

{plot_ref}.imshow(data, interpolation='none')
            ");
        }

        string byte_image_plot(string data, string plot_ref = "pl")
        {
            return string.Format($@"
data = np.array({data}) / 255.0

{plot_ref}.imshow(data, interpolation='none')
            ");
        }

        string multiplot(params string[] plots)
        {
            string pl = "";

            for (int i = 0; i < plots.Length; i++)
            {
                pl += "pl" + (i + 1) + ',';
            }

            pl = string.Format($@"
fig, ({pl}) = pl.subplots(1,{plots.Length})
            ");

            for (int i = 0; i < plots.Length; i++)
            {
                pl += "\n" + plots[i] + "\n";
            }

            return pl;
        }

        public string make_plot(float[] data, string plot_ref = "pl")
        {
            return line_plot(py_array(data), plot_ref);
        }

        public string make_plot(List<float[]> data, string plot_ref = "pl")
        {
            string s = "";
            foreach (var vec in data)
            {
                s += line_plot(py_array(vec), plot_ref) + '\n';
            }

            return s;
        }

        public string make_plot(float[,] data, string plot_ref = "pl")
        {
            return image_plot(py_array(data), plot_ref);
        }

        public string make_plot(float[,,] data, string plot_ref = "pl")
        {
            return image_plot(py_array(data), plot_ref);
        }

        public string make_plot(byte[,,] data, string plot_ref = "pl")
        {
            return byte_image_plot(py_array(data), plot_ref);
        }

        public string make_plot(object data, string plot_ref = "pl")
        {
            Type t = data.GetType();

            if      (t == typeof(float[]))   return make_plot((float[])   data, plot_ref);
            else if (t == typeof(float[,]))  return make_plot((float[,])  data, plot_ref);
            else if (t == typeof(float[,,])) return make_plot((float[,,]) data, plot_ref);
            else if (t == typeof(byte[,,]))  return make_plot((byte[,,])  data, plot_ref);

            else if (t == typeof(List<float[]>)) return make_plot((List<float[]>)data, plot_ref);

            else throw new NotImplementedException();
        }

        public string make_plot(params object[] data)
        {
            if (data.Length == 1)
            {
                return make_plot(data[0]);
            }

            string[] plots = new string[data.Length];
            for (int i = 0; i < data.Length; i++)
            {
                plots[i] = make_plot(data[i], plot_ref: "pl" + (i + 1));
            }

            return multiplot(plots);
        }

        public void plot(params object[] data)
        {
            string py = py_plot(make_plot(data));
            run_plot(py);
        }

        public void Test()
        {
            // Test line plot.
            plot(new float[] { 1, 2, 4, 9, 16, 25 });

            // Test double line plot.
            var list = new List<float[]>();
            list.Add(new float[] { 1, 2, 3 });
            list.Add(new float[] { 2, 2, 1 });
            plot(list);

            // Test image plot.
            plot(new float[,] { { 1, 2, 3, 4 }, { 4, 3, 2, 1 } });

            // Test RGB image plot.
            plot(new float[,,] { { { 1, 1, 1 }, { 1, 0, 0 } }, { { 0, 1, 0 }, { 0, 0, 1 } } });

            // Test double plot.
            plot(
                new float[] { 1, 2, 4, 9, 16, 25 },
                new float[,,] { { { 1, 1, 1 }, { 1, 0, 0 } }, { { 0, 1, 0 }, { 0, 0, 1 } } }
            );
        }
    }
}