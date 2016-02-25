using ANN.Core;
using ANN.Function;
using ANN.Learning;
using ANN.Utils;
using LumenWorks.Framework.IO.Csv;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NumericsTestBench
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
            //// Using managed code only
            //Control.UseManaged();
            //Console.WriteLine(Control.LinearAlgebraProvider);

            //var m = Matrix<double>.Build.Random(784, 30000);
            //var v = Matrix<double>.Build.Random(784, 30000);

            //var w = Stopwatch.StartNew();
            //var y1 = -(m - v).PointwiseMultiply(v).PointwiseMultiply(1 - v);
            //Console.WriteLine(w.Elapsed);
            ////Console.WriteLine(y1);

            //// Using the Intel MKL native provider
            //Control.UseNativeMKL();
            //Console.WriteLine(Control.LinearAlgebraProvider);

            //w.Restart();
            //var y2 = -(m - v).PointwiseMultiply(v).PointwiseMultiply(1 - v);
            //Console.WriteLine(w.Elapsed);
            ////Console.WriteLine(y2);

            double[][] samples = GetSamples();
            double[][] patches = GetPatches(samples, 512, 512, 10000, 8);
            patches = Maths.RemoveDcComponent(patches);
            patches = Maths.TruncateAndRescale(patches, 0.1, 0.9);

            ActivationFunction f = new SigmoidFunction();
            Layer layer1 = new Layer(25, 64, f);
            Layer layer2 = new Layer(64, 25, f);
            Network network = new Network(new Layer[] { layer1, layer2 });
            SparseAutoencoderLearning sae = new SparseAutoencoderLearning(network, 10000, 0.5, 0.0000, 0.01, 0, true);

            Stopwatch stopwatch = new Stopwatch();
            double[,] patchesArray = new double[64, 10000];
            for (int i = 0; i < 10000; i++)
            {
                for (int j = 0; j < 64; j++)
                {
                    patchesArray[j, i] = patches[i][j];
                }
            }

            Control.UseNativeMKL();
            Console.WriteLine("Start activations");
            stopwatch.Start();
            sae.UpdateCachedActivations(patches);
            Console.WriteLine("Activations computed in: {0} milliseconds", stopwatch.ElapsedMilliseconds);
            Console.WriteLine("Start derivatives");
            stopwatch.Restart();
            sae.ComputeBatchPartialDerivativesMatrix(Networks.AverageActivations(sae.Network, sae.CachedActivations), patchesArray, patchesArray);
            Console.WriteLine("Derivatives computed in: {0} milliseconds", stopwatch.ElapsedMilliseconds);
        }
    }
}
