using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ANN.Core;
using ANN.Function;

namespace ANN.Test.Core
{
    [TestClass]
    public class NetworkTest
    {
        [TestMethod]
        public void NetworkUpdate_UpdatesAllLayers()
        {
            ActivationFunction sigmoidFunction = new SigmoidFunction();
            Layer[] layers = new Layer[] { new Layer(2, 2, sigmoidFunction), new Layer(1, 2, sigmoidFunction) };
            Network network = new Network(layers);
            double[] input = new double[] { 0.5, 0.6 };
            double[] expected = new double[] { 0.626138674824 };

            layers[0][0][0] = 0.1;
            layers[0][0][1] = 0.2;
            layers[0][1][0] = 0.3;
            layers[0][1][1] = 0.4;
            layers[1][0][0] = 0.5;
            layers[1][0][1] = 0.5;
            layers[0][0].Bias = -0.01;
            layers[0][1].Bias = -0.02;
            layers[1][0].Bias = -0.05;

            double[] actual = network.Update(input);
            double[] networkOutput = network.Output;
            Assert.AreEqual(expected[0], actual[0], 0.0001, "Invalid network output");
            Assert.AreEqual(expected[0], networkOutput[0], 0.0001, "Invalid network output");
        }

        [TestMethod]
        public void ComputeNeuronOutputs_ReturnsAllNeuronOutput()
        {
            ActivationFunction sigmoidFunction = new SigmoidFunction();
            Layer[] layers = new Layer[] { new Layer(2, 2, sigmoidFunction), new Layer(1, 2, sigmoidFunction) };
            Network network = new Network(layers);
            double[] input = new double[] { 0.5, 0.6 };
            double[][] expected = new double[][]
            {
                new double[] { 0.539914884556, 0.591458978433 },
                new double[] { 0.626138674824 }
            };

            layers[0][0][0] = 0.1;
            layers[0][0][1] = 0.2;
            layers[0][1][0] = 0.3;
            layers[0][1][1] = 0.4;
            layers[1][0][0] = 0.5;
            layers[1][0][1] = 0.5;
            layers[0][0].Bias = -0.01;
            layers[0][1].Bias = -0.02;
            layers[1][0].Bias = -0.05;

            double[][] actual = network.ComputeNeuronOutputs(input);
            Assert.AreEqual(expected[0][0], actual[0][0], 0.0001, "Invalid network output");
            Assert.AreEqual(expected[0][1], actual[0][1], 0.0001, "Invalid network output");
            Assert.AreEqual(expected[1][0], actual[1][0], 0.0001, "Invalid network output");
        }
    }
}
