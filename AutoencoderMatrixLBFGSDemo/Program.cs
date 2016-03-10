using ANN.Core;
using ANN.Function;
using ANN.Learning;
using ANN.Utils;
using LumenWorks.Framework.IO.Csv;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Configuration;
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

        private static string dataSet;
        private static int dataSetSamples;
        private static int dataSetSamplesSize;

        private static int inputLayerNodes;
        private static int hiddenLayerNodes;
        private static int outputLayerNodes;

        private static double lambda;
        private static double sparsity;
        private static double beta;

        private static double lbfgsEpsg;
        private static double lbfgsEpsf;
        private static double lbfgsEpsx;
        private static int lbfgsMaxIts;
        private static int lbfgsCorrs;

        private static int filtersPerRow;
        private static int filtersWidth;
        private static int filtersHeight;
        private static int filtersPixelSize;

        private static int testSamples;
        private static int testSamplesPerRow;
        private static int testSamplesPerColumn;
        private static int testSamplesWidth;
        private static int testSamplesHeight;
        private static int testSamplesPixelSize;

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

        private static void LoadExperimentParameters()
        {
            dataSet = ConfigurationManager.AppSettings.Get("dataSet");
            dataSetSamples = int.Parse(ConfigurationManager.AppSettings.Get("dataSetSamples"));
            dataSetSamplesSize = int.Parse(ConfigurationManager.AppSettings.Get("dataSetSamplesSize"));

            inputLayerNodes = int.Parse(ConfigurationManager.AppSettings.Get("inputLayerNodes"));
            hiddenLayerNodes = int.Parse(ConfigurationManager.AppSettings.Get("hiddenLayerNodes"));
            outputLayerNodes = int.Parse(ConfigurationManager.AppSettings.Get("outputLayerNodes"));

            lambda = double.Parse(ConfigurationManager.AppSettings.Get("lambda"));
            sparsity = double.Parse(ConfigurationManager.AppSettings.Get("sparsity"));
            beta = double.Parse(ConfigurationManager.AppSettings.Get("beta"));

            lbfgsEpsg = double.Parse(ConfigurationManager.AppSettings.Get("lbfgsEpsg"));
            lbfgsEpsf = double.Parse(ConfigurationManager.AppSettings.Get("lbfgsEpsf"));
            lbfgsEpsx = double.Parse(ConfigurationManager.AppSettings.Get("lbfgsEpsx"));
            lbfgsMaxIts = int.Parse(ConfigurationManager.AppSettings.Get("lbfgsMaxIts"));
            lbfgsCorrs = int.Parse(ConfigurationManager.AppSettings.Get("lbfgsCorrs"));

            filtersPerRow = int.Parse(ConfigurationManager.AppSettings.Get("filtersPerRow"));
            filtersWidth = int.Parse(ConfigurationManager.AppSettings.Get("filtersWidth"));
            filtersHeight = int.Parse(ConfigurationManager.AppSettings.Get("filtersHeight"));
            filtersPixelSize = int.Parse(ConfigurationManager.AppSettings.Get("filtersPixelSize"));

            testSamples = int.Parse(ConfigurationManager.AppSettings.Get("testSamples"));
            testSamplesPerRow = int.Parse(ConfigurationManager.AppSettings.Get("testSamplesPerRow"));
            testSamplesPerColumn = int.Parse(ConfigurationManager.AppSettings.Get("testSamplesPerColumn"));
            testSamplesWidth = int.Parse(ConfigurationManager.AppSettings.Get("testSamplesWidth"));
            testSamplesHeight = int.Parse(ConfigurationManager.AppSettings.Get("testSamplesHeight"));
            testSamplesPixelSize = int.Parse(ConfigurationManager.AppSettings.Get("testSamplesPixelSize"));
        }

        private static void PrintExperimentParameters()
        {
            Console.WriteLine("Data set: {0}", dataSet);
            Console.WriteLine("Data set samples: {0}", dataSetSamples);
            Console.WriteLine("Data set samples size: {0}", dataSetSamplesSize);
            Console.WriteLine();
            Console.WriteLine("Network parameters");
            Console.WriteLine("Input layer nodes: {0}", inputLayerNodes);
            Console.WriteLine("Hidden layer nodes: {0}", hiddenLayerNodes);
            Console.WriteLine("Output layer nodes: {0}", outputLayerNodes);
            Console.WriteLine();
            Console.WriteLine("Sparse autoencoder parameters");
            Console.WriteLine("Lambda: {0}", lambda);
            Console.WriteLine("Sparsity: {0}", sparsity);
            Console.WriteLine("Beta: {0}", beta);
            Console.WriteLine();
            Console.WriteLine("L-BFGS parameters");
            Console.WriteLine("Epsg: {0}", lbfgsEpsg);
            Console.WriteLine("Epsf: {0}", lbfgsEpsf);
            Console.WriteLine("Epsx: {0}", lbfgsEpsx);
            Console.WriteLine("MaxIts: {0}", lbfgsMaxIts);
            Console.WriteLine("Corrs: {0}", lbfgsCorrs);
            Console.WriteLine();
            Console.WriteLine("Filter export parameters");
            Console.WriteLine("Filters per row: {0}", filtersPerRow);
            Console.WriteLine("Filters width: {0}", filtersWidth);
            Console.WriteLine("Filters height: {0}", filtersHeight);
            Console.WriteLine("Filters pixel size: {0}", filtersPixelSize);
            Console.WriteLine();
            Console.WriteLine("Test export parameters");
            Console.WriteLine("Test samples: {0}", testSamples);
            Console.WriteLine("Test samples per row: {0}", testSamplesPerRow);
            Console.WriteLine("Test samples per column: {0}", testSamplesPerColumn);
            Console.WriteLine("Test samples width: {0}", testSamplesWidth);
            Console.WriteLine("Test samples height: {0}", testSamplesHeight);
            Console.WriteLine("Test samples pixel size: {0}", testSamplesPixelSize);
            Console.WriteLine();
        }

        public static void Main(string[] args)
        {
            LoadExperimentParameters();
            PrintExperimentParameters();

            Console.WriteLine("Press any key to run this experiment...");
            Console.ReadKey();

            Console.WriteLine("Loading train data");
            Control.UseNativeMKL();
            Matrix<double> input = (dataSet == "mnist" ? GetMnistTrainSamples(dataSetSamples) : GetPatchesTrainSamples(dataSetSamples, dataSetSamplesSize));

            Console.WriteLine("Initializing neural network");
            ActivationFunction f = new SigmoidFunction();
            Layer layer1 = new Layer(hiddenLayerNodes, inputLayerNodes, f);
            Layer layer2 = new Layer(outputLayerNodes, hiddenLayerNodes, f);
            Network network = new Network(new Layer[] { layer1, layer2 });
            SparseAutoencoderMatrixLearning saem = new SparseAutoencoderMatrixLearning(network, lambda, sparsity, beta);
            SparseAutoencoderMatrixAdapter saemAdapter = new SparseAutoencoderMatrixAdapter(saem, input, input);

            Console.WriteLine("Initializing L-BFGS parameters");
            double[] x = saem.ParametersArray();
            alglib.minlbfgsstate state;
            alglib.minlbfgsreport rep;

            Console.WriteLine("Begin training and optimisation");
            alglib.minlbfgscreate(lbfgsCorrs, x, out state);
            alglib.minlbfgssetcond(state, lbfgsEpsg, lbfgsEpsf, lbfgsEpsx, lbfgsMaxIts);
            alglib.minlbfgssetxrep(state, true);
            alglib.minlbfgsoptimize(state, saemAdapter.FunctionValueAndGradient, saemAdapter.PrintProgress, null);
            alglib.minlbfgsresults(state, out x, out rep);

            Console.WriteLine("{0}", rep.terminationtype);
            //Console.WriteLine("{0}", alglib.ap.format(x, 2));

            Console.WriteLine("Updating underlying network");
            saem.UpdateUnderlyingNetwork();

            Console.WriteLine("Exporting network parameters");
            Networks.ExportParametersToText(network);

            Console.WriteLine("Exporting filters");
            Networks.ExportFiltersToBitmap(network, filtersPerRow, filtersWidth, filtersHeight, filtersPixelSize);

            Console.WriteLine("Loading test data");
            Matrix<double> test = (dataSet == "mnist" ? GetMnistTestSamples(testSamples) : GetPatchesTrainSamples(testSamples, dataSetSamplesSize));

            Console.WriteLine("Exporting reconstructions");
            Networks.ExportReconstructionsToBitmap(network, test, testSamplesPerRow, testSamplesPerColumn, testSamplesWidth, testSamplesHeight, testSamplesPixelSize);

            Console.WriteLine("Experiment complete");
            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
        }
    }
}
