using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ANN.Function;

namespace ANN.Test.Function
{
    [TestClass]
    public class SquaredErrorCostFunctionTest
    {
        [TestMethod]
        public void SquaredErrorVectorCost_ReturnsVectorCost()
        {
            double[] output = new double[] { 0.2, 0.8, 1.2 };
            double[] target = new double[] { 0, 0.7, 1.5 };
            double expected = 0.0233333333334;
            double actual = CostFunctions.HalfSquaredError(output, target);

            Assert.AreEqual(expected, actual, 0.0001, "Invalid squared error cost");
        }

        [TestMethod]
        public void SquaredErrorCost_ReturnsCost()
        {
            double output = 1.2;
            double target = 1;
            double expected = 0.02;
            double actual = CostFunctions.HalfSquaredError(output, target);

            Assert.AreEqual(expected, actual, 0.0001, "Invalid squared error cost");
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void SquaredErrorVectorCost_WhenInvalidVectors_ShouldThrowArgument()
        {
            double[] output = new double[] { 0.2, 0.8, 1.2 };
            double[] target = new double[] { 0, 0.7 };
            double actual = CostFunctions.HalfSquaredError(output, target);
        }
    }
}
