using ANN.Utils;
using LumenWorks.Framework.IO.Csv;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SamplesTestBench
{
    public class Program
    {
        public static Random rng = new Random();

        private static double[][] GetSamples()
        {
            List<double[]> samples = new List<double[]>();

            using (CsvReader csv =
                   new CsvReader(new StringReader(Properties.Resources.images), false))
            {
                int fieldCount = csv.FieldCount;
                Console.WriteLine("Field count: {0}", fieldCount);

                while (csv.ReadNextRecord())
                {
                    List<double> trainingExample = new List<double>();

                    for (int i = 0; i < fieldCount; i++)
                    {
                        trainingExample.Add(double.Parse(csv[i]));
                    }

                    samples.Add(trainingExample.ToArray());
                }
            }

            Console.WriteLine("Total examples: {0}", samples.Count);
            return samples.ToArray();
        }

        private static double[][] GetPatches(double[][] samples, int samplesWidth, int samplesHeight, int patchesNum, int patchesSize)
        {
            List<double[]> patches = new List<double[]>();
            int patchesPerSample = patchesNum / samples.Length;
            double[] patch;

            for (int i = 0; i < samples.Length; i++)
            {
                for (int j = 0; j < patchesPerSample; j++)
                {
                    patch = new double[patchesSize * patchesSize];
                    int leftCornerWidth = rng.Next(0, samplesWidth - patchesSize);
                    int leftCornerHeight = rng.Next(0, samplesHeight - patchesSize);
                    int leftCorner = leftCornerHeight * samplesWidth + leftCornerWidth;

                    for (int h = 0; h < patchesSize; h++)
                    {
                        for (int w = 0; w < patchesSize; w++)
                        {
                            patch[h * patchesSize + w] = samples[i][leftCorner + w + h * samplesWidth];
                        }
                    }

                    patches.Add(patch);
                }
            }

            return patches.ToArray();
        }

        public static void ExportToBitmap(double[][] pixelValues, int width, int height, int wdiv, int hdiv, string basefilename)
        {
            double[] normalisedValues;
            int[] intNormalisedValues;
            int wstep = width / wdiv;
            int hstep = height / hdiv;
            string filename;
            int value;
            Bitmap bmp;

            for (int i = 0; i < pixelValues.Length; i++)
            {
                Console.WriteLine("Exporting sample {0}", i);
                bmp = new Bitmap(width, height);

                Console.WriteLine("Rescaling");
                normalisedValues = Maths.Rescale(pixelValues[i], 0, 1);
                Console.WriteLine("Scaling to grey values");
                intNormalisedValues = normalisedValues.Select(n => (int)(n * 255)).ToArray();

                Console.WriteLine("Drawing");
                using (Graphics g = Graphics.FromImage(bmp))
                {
                    for (int h = 0; h < hdiv; h++)
                    {
                        for (int w = 0; w < wdiv; w++)
                        {
                            value = intNormalisedValues[w + h * wdiv];
                            g.FillRectangle(new SolidBrush(Color.FromArgb(value, value, value)),
                                new Rectangle(w * wstep, h * hstep, wstep, hstep));
                        }
                    }
                }

                Console.WriteLine("Saving");
                filename = string.Format("{0}{1}.bmp", basefilename, i);
                bmp.Save(filename);
            }
        }

        public static void Main(string[] args)
        {
            double[][] samples = GetSamples();

            //ExportToBitmap(samples, 512, 512, 512, 512, "sample");

            double[][] patches = GetPatches(samples, 512, 512, 100, 8);

            ExportToBitmap(patches, 8, 8, 8, 8, "patch");

            patches = Maths.RemoveDcComponent(patches);

            ExportToBitmap(patches, 8, 8, 8, 8, "patchdc");

            patches = Maths.TruncateAndRescale(patches, 0.1, 0.9);

            ExportToBitmap(patches, 8, 8, 8, 8, "patchdctruncate");
        }
    }
}
