using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ANN.Function;

namespace ANN.Test.Function
{
    [TestClass]
    public class ThresholdFunctionTest
    {
        [TestMethod]
        public void ThresholdOutput_ReturnsStepValue()
        {
            ActivationFunction thresholdFunction = new ThresholdFunction();
            double input1 = -1;
            double input2 = 0;
            double input3 = 1;
            double expected1 = 0;
            double expected2 = 1;
            double expected3 = 1;
            double actual1 = thresholdFunction.Output(input1);
            double actual2 = thresholdFunction.Output(input2);
            double actual3 = thresholdFunction.Output(input3);

            Assert.AreEqual(expected1, actual1, 0.0001, "Invalid threshold output");
            Assert.AreEqual(expected2, actual2, 0.0001, "Invalid threshold output");
            Assert.AreEqual(expected3, actual3, 0.0001, "Invalid threshold output");
        }
    }
}
