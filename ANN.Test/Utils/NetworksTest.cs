using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ANN.Core;
using ANN.Function;
using ANN.Utils;

namespace ANN.Test.Utils
{
    [TestClass]
    public class NetworksTest
    {
        [TestMethod]
        public void WeightsSum_ReturnsSum()
        {
            ActivationFunction sigmoidFunction = new SigmoidFunction();
            Layer[] layers = new Layer[] { new Layer(3, 2, sigmoidFunction), new Layer(2, 3, sigmoidFunction) };
            Network network = new Network(layers);

            layers[0][0][0] = 0.1;
            layers[0][0][1] = 0.2;
            layers[0][1][0] = 0.3;
            layers[0][1][1] = 0.4;
            layers[0][2][0] = 0.5;
            layers[0][2][1] = 0.6;
            layers[1][0][0] = 0.5;
            layers[1][0][1] = 0.5;
            layers[1][0][2] = 0.5;
            layers[1][1][0] = 0.6;
            layers[1][1][1] = 0.6;
            layers[1][1][2] = 0.6;
            layers[0][0].Bias = -0.01;
            layers[0][1].Bias = -0.02;
            layers[1][0].Bias = -0.05;

            double actual = Networks.WeightsSum(network);

            Assert.AreEqual(5.4, actual, 0.0001, "Invalid weights sum");
        }

        [TestMethod]
        public void AverageActivations_ReturnsAverages()
        {
            ActivationFunction sigmoidFunction = new SigmoidFunction();
            Layer[] layers = new Layer[] { new Layer(3, 2, sigmoidFunction), new Layer(2, 3, sigmoidFunction) };
            Network network = new Network(layers);

            double[][][] activations = new double[][][]
            {
                new double[][]
                {
                    new double[] { 1.2, 1.3, 1.4 },
                    new double[] { 0.2, 0.3 }
                },
                new double[][]
                {
                    new double[] { 1.0, 1.7, 0.4 },
                    new double[] { 0.5, 0.1 }
                },
                new double[][]
                {
                    new double[] { 1.4, 0.9, 0.1 },
                    new double[] { 1.1, 0.9 }
                },
                new double[][]
                {
                    new double[] { 1.4, 1.3, 1.2 },
                    new double[] { 0.5, 0.6 }
                }
            };

            double[][] expected = new double[][]
            {
                new double[] { 1.25, 1.3, 0.775 },
                new double[] { 0.575, 0.475 }
            };

            double[][] actual = Networks.AverageActivations(network, activations);

            for (int i = 0; i < expected.Length; i++)
            {
                for (int j = 0; j < expected[i].Length; j++)
                {
                    Assert.AreEqual(expected[i][j], actual[i][j], 0.0001, "Incorrect average activations");
                }
            }
        }

        [TestMethod]
        public void AverageActivations_ReturnsZeros()
        {
            ActivationFunction sigmoidFunction = new SigmoidFunction();
            Layer[] layers = new Layer[] { new Layer(3, 2, sigmoidFunction), new Layer(2, 3, sigmoidFunction) };
            Network network = new Network(layers);

            double[][][] activations = new double[0][][];

            double[][] expected = new double[][]
            {
                new double[] { 0, 0, 0 },
                new double[] { 0, 0 }
            };

            double[][] actual = Networks.AverageActivations(network, activations);

            for (int i = 0; i < expected.Length; i++)
            {
                for (int j = 0; j < expected[i].Length; j++)
                {
                    Assert.AreEqual(expected[i][j], actual[i][j], 0.0001, "Incorrect average activations");
                }
            }
        }
    }
}
