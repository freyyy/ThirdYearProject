using ANN.Core;
using ANN.Function;
using ANN.Learning;
using ANN.Utils;
using LumenWorks.Framework.IO.Csv;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace AutoencoderLBFGSDemo
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

        public static void Main(string[] args)
        {
            Console.WriteLine("Vector hardware acceleration enabled: {0}", Vector.IsHardwareAccelerated);
            double[][] samples = GetSamples();
            double[][] patches = GetPatches(samples, 512, 512, 10000, 8);
            patches = Maths.RemoveDcComponent(patches);
            patches = Maths.TruncateAndRescale(patches, 0.1, 0.9);

            ActivationFunction f = new SigmoidFunction();
            Layer layer1 = new Layer(25, 64, f);
            Layer layer2 = new Layer(64, 25, f);
            Network network = new Network(new Layer[] { layer1, layer2 });
            SparseAutoencoderLearning sae = new SparseAutoencoderLearning(network, 10000, 1, 0.0002, 0.02, 6);
            SparseAutoencoderAdapter saeAdapter = new SparseAutoencoderAdapter(sae, patches);

            double[] x = saeAdapter.GetFunctionParameters();
            double epsg = 0.0000000001;
            double epsf = 0;
            double epsx = 0;
            int maxits = 400;
            alglib.minlbfgsstate state;
            alglib.minlbfgsreport rep;

            alglib.minlbfgscreate(5, x, out state);
            alglib.minlbfgssetcond(state, epsg, epsf, epsx, maxits);
            alglib.minlbfgssetxrep(state, true);
            alglib.minlbfgsoptimize(state, saeAdapter.FunctionValueAndGradient, saeAdapter.PrintProgress, null);
            alglib.minlbfgsresults(state, out x, out rep);

            Console.WriteLine("{0}", rep.terminationtype);
            //Console.WriteLine("{0}", alglib.ap.format(x, 2));

            Networks.ExportHiddenWeightsToBitmap(network, 128, 128, 8, 8);
        }
    }
}
