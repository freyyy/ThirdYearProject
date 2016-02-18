using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ANN.Function;
using ANN.Core;
using ANN.Learning;
using ANN.Utils;

namespace ANN.Test.Learning
{
    [TestClass]
    public class SparseAutoencoderAdapterTest
    {
        [TestMethod]
        public void FunctionValueAndGradient_UpdatesGradientsAndValue()
        {
            ActivationFunction sigmoidFunction = new SigmoidFunction();
            Layer[] layers = new Layer[] { new Layer(2, 2, sigmoidFunction), new Layer(2, 2, sigmoidFunction) };
            Network network = new Network(layers);
            SparseAutoencoderLearning sparseAutoencoder = new SparseAutoencoderLearning(network, 3, 0.5, 0.0001, 0.1, 3);
            double[][] input = new double[][] { new double[] { 0.5, 0.6 }, new double[] { 0.1, 0.2 }, new double[] { 0.3, 0.3 } };
            double[][] output = new double[input.Length][];

            SparseAutoencoderAdapter sparseAutoencoderAdapter = new SparseAutoencoderAdapter(sparseAutoencoder, input);
            int index = 0;
            double func = 0;
            double[] x = new double[12];
            double[] grad = new double[12];

            for (int i = 0; i < input.Length; i++)
            {
                output[i] = network.Update(input[i]);
            }

            Tuple<double[][][], double[][]> expectedGradients = sparseAutoencoder.ComputeBatchPartialDerivatives(input, input);
            Tuple<double[][][], double[][]> expectedGradients2 = sparseAutoencoder.ComputeBatchPartialDerivatives(input, input);
            double[][] averageActivations = Networks.AverageActivations(network, sparseAutoencoder.CachedActivations);
            double expectedValue = CostFunctions.HalfSquaredErrorL2Sparsity(network, averageActivations[0], sparseAutoencoder.Sparsity, sparseAutoencoder.Lambda, sparseAutoencoder.Beta, output, input);

            for (int i = 0; i < network.LayerCount; i++)
            {
                for (int j = 0; j < network[i].NeuronCount; j++)
                {
                    x[index++] = network[i][j].Bias;
                    network[i][j].Bias = 0;

                    for (int k = 0; k < network[i][j].InputCount; k++)
                    {
                        x[index++] = network[i][j][k];
                        network[i][j][k] = 0;
                    }
                }
            }

            index = 0;

            sparseAutoencoderAdapter.FunctionValueAndGradient(x, ref func, grad, null);

            for (int i = 0; i < expectedGradients.Item1.Length; i++)
            {
                for (int j = 0; j < expectedGradients.Item1[i].Length; j++)
                {
                    Assert.AreEqual(expectedGradients.Item2[i][j], grad[index++], 0.0001, "Gradient updated incorrectly {0} {1}", i, j);

                    for (int k = 0; k < expectedGradients.Item1[i][j].Length; k++)
                    {
                        Assert.AreEqual(expectedGradients.Item1[i][j][k], grad[index++], 0.0001, "Gradient updated incorrectly {0} {1} {2}", i, j, k);
                    }
                }
            }

            Assert.AreEqual(expectedValue, func, 0.0001, "Value updated incorrectly");
        }

        [TestMethod]
        public void GetFunctionParameters_ReturnsParameters()
        {
            ActivationFunction sigmoidFunction = new SigmoidFunction();
            Layer[] layers = new Layer[] { new Layer(2, 2, sigmoidFunction), new Layer(2, 2, sigmoidFunction) };
            Network network = new Network(layers);
            SparseAutoencoderLearning sparseAutoencoder = new SparseAutoencoderLearning(network, 3, 0.5, 0.0001, 0.1, 3);
            double[][] input = new double[][] { new double[] { 0.5, 0.6 }, new double[] { 0.1, 0.2 }, new double[] { 0.3, 0.3 } };
            double[][] output = new double[input.Length][];

            SparseAutoencoderAdapter sparseAutoencoderAdapter = new SparseAutoencoderAdapter(sparseAutoencoder, input);

            double[] parameters = sparseAutoencoderAdapter.GetFunctionParameters();
            int index = 0;

            Assert.AreEqual(12, parameters.Length, 0, "Invalid parameters length");

            for (int i = 0; i < network.LayerCount; i++)
            {
                for (int j = 0; j < network[i].NeuronCount; j++)
                {
                    Assert.AreEqual(network[i][j].Bias, parameters[index++], 0.0001, "Invalid bias parameter");

                    for (int k = 0; k < network[i][j].InputCount; k++)
                    {
                        Assert.AreEqual(network[i][j][k], parameters[index++], 0.0001, "Invalid weight parameter");
                    }
                }
            }
        }
    }
}
