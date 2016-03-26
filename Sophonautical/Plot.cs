using System;
using System.IO;
using System.Text;
using System.Diagnostics;

namespace Sophonautical
{
    class Plot
    {
        string OutputDir;

        public Plot(string OutputDir)
        {
            this.OutputDir = OutputDir;
        }

        void run_cmd(string cmd, string args)
        {
            ProcessStartInfo start = new ProcessStartInfo();
            start.FileName = "python";
            start.Arguments = string.Format("{0} {1}", cmd, args);
            start.UseShellExecute = false;
            start.RedirectStandardOutput = true;

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
            string py_file = Path.Combine(OutputDir, "plot.py");

            File.WriteAllText(py_file, py);

            run_cmd(py_file, "");
        }

        string py_array(float[] data)
        {
            return '[' + string.Join(",", data) + ']';
        }

        string py_array(float[,] data)
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

        string py_array<T>(T[,,] data)
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

        string py_plot(string plot)
        {
            return string.Format($@"
import numpy as np
import matplotlib.pyplot as pl

{plot}

pl.show()
            ");
        }

        string line_plot(string data, string plot_ref = "pl")
        {
            return string.Format($@"
data = np.array({data})

{plot_ref}.plot(data)
            ");
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

        string multiplot(string plot1, string plot2)
        {
            return string.Format($@"
fig, (pl1, pl2) = pl.subplots(1,2)

{plot1}

{plot2}
            ");
        }

        public string make_plot(float[] data, string plot_ref = "pl")
        {
            return line_plot(py_array(data), plot_ref);
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
            else throw new NotImplementedException();
        }

        public string make_plot(object data1, object data2)
        {
            return multiplot(make_plot(data1, plot_ref:"pl1"), make_plot(data2, plot_ref:"pl2"));
        }

        public void plot(object data)
        {
            string py = py_plot(make_plot(data));
            run_plot(py);
        }

        public void plot(object data1, object data2)
        {
            string py = py_plot(make_plot(data1, data2));
            run_plot(py);
        }

        public void Test()
        {
            // Test line plot.
            plot(new float[] { 1, 2, 4, 9, 16, 25 });

            // Test image plot.
            plot(new float[,] { { 1, 2, 3, 4 }, { 4, 3, 2, 1 } });

            // Test RGB image plot.
            plot(new float[,,] { { { 1, 1, 1 }, { 1, 0, 0 } }, { { 0, 1, 0 }, { 0, 0, 1 } } });
        }
    }
}