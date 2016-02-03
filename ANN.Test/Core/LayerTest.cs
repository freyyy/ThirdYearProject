using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ANN.Core;
using ANN.Function;

namespace ANN.Test.Core
{
    [TestClass]
    public class LayerTest
    {
        [TestMethod]
        public void LayerUpdate_UpdatesAllNeurons()
        {
            ActivationFunction sigmoidFunction = new SigmoidFunction();
            Layer layer = new Layer(2, 2, sigmoidFunction);
            double[] input = new double[] { 0.5, 0.6 };
            double[] expected = new double[] { 0.539914884556, 0.591458978433 };

            layer[0][0] = 0.1;
            layer[0][1] = 0.2;
            layer[1][0] = 0.3;
            layer[1][1] = 0.4;
            layer[0].Bias = -0.01;
            layer[1].Bias = -0.02;

            double[] actual = layer.Update(input);
            double[] layerOutput = layer.Output;
            Assert.AreEqual(expected[0], actual[0], 0.0001, "Invalid layer output");
            Assert.AreEqual(expected[0], layerOutput[0], 0.0001, "Invalid layer output");
            Assert.AreEqual(expected[1], actual[1], 0.0001, "Invalid layer output");
            Assert.AreEqual(expected[1], layerOutput[1], 0.0001, "Invalid layer output");
        }
    }
}
