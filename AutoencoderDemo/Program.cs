using ANN.Core;
using ANN.Function;
using ANN.Learning;
using ANN.Utils;
using LumenWorks.Framework.IO.Csv;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace AutoencoderDemo
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

                    for (int w = 0; w < patchesSize; w++)
                    {
                        for (int h = 0; h < patchesSize; h++)
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
            double[][] input = new double[][]
            {
                new double[] { 1, 1, 1, 0, 0, 0 },
                new double[] { 1, 0, 1, 0, 0, 0 },
                new double[] { 1, 1, 1, 0, 0, 0 },
                new double[] { 0, 0, 1, 1, 1, 0 },
                new double[] { 0, 0, 1, 0, 1, 0 },
                new double[] { 0, 0, 1, 1, 1, 0 },
            };
            double lastError = 1;
            double error = 1;
            double delta = 0.1;
            int epoch = 0;

            ActivationFunction f = new SigmoidFunction();
            Layer layer1 = new Layer(2, 6, f);
            Layer layer2 = new Layer(6, 2, f);
            Network network = new Network(new Layer[] { layer1, layer2 });
            SparseAutoencoderLearning sae = new SparseAutoencoderLearning(network, 6, 1, 0.0001, 0.5, 0.1);

            Console.WriteLine("Hidden unit 1 weights: {0} {1} {2} {3} {4} {5} {6}", network[0][0].Bias, network[0][0][0], network[0][0][1], network[0][0][2], network[0][0][3], network[0][0][4], network[0][0][5]);
            Console.WriteLine("Hidden unit 2 weights: {0} {1} {2} {3} {4} {5} {6}", network[0][1].Bias, network[0][1][0], network[0][1][1], network[0][1][2], network[0][1][3], network[0][1][4], network[0][1][5]);

            while (epoch < 0)
            {
                error = sae.RunEpoch(input);
                epoch++;
                Console.WriteLine("Error after {0} epochs: {1}", epoch, error);
                delta = lastError - error;
                lastError = error;
            }
            Console.WriteLine("Training complete after {0} epochs using the Sparse Autoencoder training regime.", epoch);
            Console.WriteLine("Testing");

            Console.WriteLine("Hidden unit 1 weights: {0} {1} {2} {3} {4} {5} {6}", network[0][0].Bias, network[0][0][0], network[0][0][1], network[0][0][2], network[0][0][3], network[0][0][4], network[0][0][5]);
            Console.WriteLine("Hidden unit 2 weights: {0} {1} {2} {3} {4} {5} {6}", network[0][1].Bias, network[0][1][0], network[0][1][1], network[0][1][2], network[0][1][3], network[0][1][4], network[0][1][5]);
            network.Update(new double[] { 1, 1, 0, 0, 0, 0 });
            Console.WriteLine("{0} {1}", network[0][0].Output, network[0][1].Output);
            network.Update(new double[] { 0, 0, 0, 1, 1, 0 });
            Console.WriteLine("{0} {1}", network[0][0].Output, network[0][1].Output);
            network.Update(new double[] { 0, 0, 0, 1, 1, 0 });
            Console.WriteLine("{0} {1}", network[0][0].Output, network[0][1].Output);

            Networks.ExportHiddenWeightsToBitmap(network, 600, 100, 6, 1);

            double[][] samples = GetSamples();
            double[][] patches = GetPatches(samples, 512, 512, 10000, 8);
        }
    }
}
