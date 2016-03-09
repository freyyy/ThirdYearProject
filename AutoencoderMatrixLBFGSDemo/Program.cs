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

namespace AutoencoderMatrixLBFGSDemo
{
    public class Program
    {
        public static Random rng = new Random();
        public static Stopwatch s = new Stopwatch();

        private static Matrix<double> GetPatchesTrainSamples(int count, int size)
        {
            List<double[]> samples = new List<double[]>();
            List<double[]> patches = new List<double[]>();
            double[] patch;

            using (CsvReader csv =
                   new CsvReader(new StringReader(Properties.Resources.images), false))
            {
                int fieldCount = csv.FieldCount;

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

            for (int i = 0; i < count; i++)
            {
                patch = new double[size * size];
                int sample = rng.Next(0, samples.Count);
                int leftCornerWidth = rng.Next(0, 512 - size);
                int leftCornerHeight = rng.Next(0, 512 - size);
                int leftCorner = leftCornerHeight * 512 + leftCornerWidth;

                for (int h = 0; h < size; h++)
                {
                    for (int w = 0; w < size; w++)
                    {
                        patch[h * size + w] = samples[sample][leftCorner + w + h * 512];
                    }
                }

                patches.Add(patch);
            }

            patches = Maths.RemoveDcComponent(patches.ToArray()).ToList();
            patches = Maths.TruncateAndRescale(patches.ToArray(), 0.1, 0.9).ToList();

            return Matrix<double>.Build.DenseOfColumnArrays(patches);
        }

        private static Matrix<double> GetMnistTrainSamples(int count)
        {
            List<double[]> samples = new List<double[]>();

            using (CsvReader csv =
                   new CsvReader(new StringReader(Properties.Resources.mnist_train), false))
            {
                int fieldCount = csv.FieldCount;

                while (csv.ReadNextRecord())
                {
                    List<double> trainingExample = new List<double>();

                    for (int i = 1; i < fieldCount; i++)
                    {
                        trainingExample.Add(double.Parse(csv[i]) / 255);
                    }

                    samples.Add(trainingExample.ToArray());
                }
            }

            samples = samples.Take(count).ToList();

            return Matrix<double>.Build.DenseOfColumnArrays(samples);
        }

        private static Matrix<double> GetMnistTestSamples(int count)
        {
            List<double[]> samples = new List<double[]>();

            using (CsvReader csv =
                   new CsvReader(new StringReader(Properties.Resources.mnist_train), false))
            {
                int fieldCount = csv.FieldCount;

                while (csv.ReadNextRecord())
                {
                    List<double> trainingExample = new List<double>();

                    for (int i = 1; i < fieldCount; i++)
                    {
                        trainingExample.Add(double.Parse(csv[i]) / 255);
                    }

                    samples.Add(trainingExample.ToArray());
                }
            }

            samples.Reverse();
            samples = samples.Take(count).ToList();

            return Matrix<double>.Build.DenseOfColumnArrays(samples);
        }

        public static void Main(string[] args)
        {
            Control.UseNativeMKL();
            Console.WriteLine("Setting up experiment");

            Matrix<double> input = GetMnistTrainSamples(100);

            ActivationFunction f = new SigmoidFunction();
            Layer layer1 = new Layer(196, 784, f);
            Layer layer2 = new Layer(784, 196, f);
            Network network = new Network(new Layer[] { layer1, layer2 });
            SparseAutoencoderMatrixLearning saem = new SparseAutoencoderMatrixLearning(network, 0.003, 0.1, 3);

            SparseAutoencoderMatrixAdapter saemAdapter = new SparseAutoencoderMatrixAdapter(saem, input, input);

            double[] x = saem.ParametersArray();
            double epsg = 0.0000000001;
            double epsf = 0;
            double epsx = 0;
            int maxits = 400;
            alglib.minlbfgsstate state;
            alglib.minlbfgsreport rep;

            alglib.minlbfgscreate(5, x, out state);
            alglib.minlbfgssetcond(state, epsg, epsf, epsx, maxits);
            alglib.minlbfgssetxrep(state, true);
            alglib.minlbfgsoptimize(state, saemAdapter.FunctionValueAndGradient, saemAdapter.PrintProgress, null);
            alglib.minlbfgsresults(state, out x, out rep);

            Console.WriteLine("{0}", rep.terminationtype);
            //Console.WriteLine("{0}", alglib.ap.format(x, 2));

            saem.UpdateUnderlyingNetwork();
            Networks.ExportFiltersToBitmap(network, 14, 28, 28, 1);

            Matrix<double> test = GetMnistTestSamples(100);
            Networks.ExportReconstructionsToBitmap(network, test, 10, 10, 28, 28, 1);
        }
    }
}
