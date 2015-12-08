using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ANN.Function;

namespace ANN.Test.Function
{
    [TestClass]
    public class SigmoidFunctionTest
    {
        [TestMethod]
        public void SigmoidOutput_ReturnsValueOfSigmoid()
        {
            ActivationFunction sigmoidFunction = new SigmoidFunction();
            double input1 = -3;
            double input2 = 0;
            double input3 = 2;
            double expected1 = 0.0474258731776;
            double expected2 = 0.5;
            double expected3 = 0.880797077978;
            double actual1 = sigmoidFunction.Output(input1);
            double actual2 = sigmoidFunction.Output(input2);
            double actual3 = sigmoidFunction.Output(input3);

            Assert.AreEqual(expected1, actual1, 0.0001, "Invalid sigmoid output");
            Assert.AreEqual(expected2, actual2, 0.0001, "Invalid sigmoid output");
            Assert.AreEqual(expected3, actual3, 0.0001, "Invalid sigmoid output");
        }

        [TestMethod]
        public void SigmoidDerivative_ReturnsSigmoidDerivative()
        {
            ActivationFunction sigmoidFunction = new SigmoidFunction();
            double input1 = -3;
            double input2 = 0;
            double input3 = 2;
            double expected1 = 0.04517665973;
            double expected2 = 0.25;
            double expected3 = 0.1049935854;
            double actual1 = sigmoidFunction.Derivative(input1);
            double actual2 = sigmoidFunction.Derivative(input2);
            double actual3 = sigmoidFunction.Derivative(input3);

            Assert.AreEqual(expected1, actual1, 0.0001, "Invalid sigmoid derivative output");
            Assert.AreEqual(expected2, actual2, 0.0001, "Invalid sigmoid derivative output");
            Assert.AreEqual(expected3, actual3, 0.0001, "Invalid sigmoid derivative output");
        }

        [TestMethod]
        public void SigmoidDerivativeOutput_ReturnsSigmoidDerivativeGivenSigmoidOutput()
        {
            ActivationFunction sigmoidFunction = new SigmoidFunction();
            double input1 = 0.0474258731776;
            double input2 = 0.5;
            double input3 = 0.880797077978;
            double expected1 = 0.04517665973;
            double expected2 = 0.25;
            double expected3 = 0.1049935854;
            double actual1 = sigmoidFunction.OutputDerivative(input1);
            double actual2 = sigmoidFunction.OutputDerivative(input2);
            double actual3 = sigmoidFunction.OutputDerivative(input3);

            Assert.AreEqual(expected1, actual1, 0.0001, "Invalid sigmoid derivative output");
            Assert.AreEqual(expected2, actual2, 0.0001, "Invalid sigmoid derivative output");
            Assert.AreEqual(expected3, actual3, 0.0001, "Invalid sigmoid derivative output");
        }
    }
}
