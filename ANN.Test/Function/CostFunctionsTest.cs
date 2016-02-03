using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ANN.Function;
using ANN.Core;

namespace ANN.Test.Function
{
    [TestClass]
    public class CostFunctionsTest
    {
        [TestMethod]
        public void HalfSquaredErrorL2Cost_ReturnsCost()
        {
            ActivationFunction sigmoidFunction = new SigmoidFunction();
            Layer[] layers = new Layer[] { new Layer(2, 2, sigmoidFunction), new Layer(1, 2, sigmoidFunction) };
            Network network = new Network(layers);

            double lambda = 0.25;
            double[][] output = new double[][]
            {
                new double[] { 0.2 },
                new double[] { 0.8 },
                new double[] { 1.2 }
            };
            double[][] target = new double[][]
            {
                new double[] { 0 },
                new double[] { 0.7 },
                new double[] { 1.5 }
            };
            double expected = 0.14667;

            layers[0][0][0] = 0.1;
            layers[0][0][1] = 0.2;
            layers[0][1][0] = 0.3;
            layers[0][1][1] = 0.4;
            layers[1][0][0] = 0.5;
            layers[1][0][1] = 0.5;

            double actual = CostFunctions.HalfSquaredErrorL2(network, lambda, output, target);

            Assert.AreEqual(expected, actual, 0.0001, "Invalid squared error cost");
        }

        [TestMethod]
        public void HalfSquaredErrorCost1D_ReturnsCost()
        {
            double[] output = new double[] { 0.2, 0.8, 1.2 };
            double[] target = new double[] { 0, 0.7, 1.5 };
            double expected = 0.07;
            double actual = CostFunctions.HalfSquaredError(output, target);

            Assert.AreEqual(expected, actual, 0.0001, "Invalid squared error cost");
        }

        [TestMethod]
        public void HalfSquaredErrorCost_ReturnsCost()
        {
            double output = 1.2;
            double target = 1;
            double expected = 0.02;
            double actual = CostFunctions.HalfSquaredError(output, target);

            Assert.AreEqual(expected, actual, 0.0001, "Invalid squared error cost");
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void HalfSquaredErrorCost_ThrowsArgument()
        {
            double[] output = new double[] { 0.2, 0.8, 1.2 };
            double[] target = new double[] { 0, 0.7 };
            double actual = CostFunctions.HalfSquaredError(output, target);
        }
    }
}
