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
    }
}
