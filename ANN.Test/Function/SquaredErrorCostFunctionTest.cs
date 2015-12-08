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
            CostFunction squaredErrorCost = new SquaredErrorCostFunction();
            double[] output = new double[] { 0.2, 0.8, 1.2 };
            double[] target = new double[] { 0, 0.7, 1.5 };
            double expected = 0.0233333333334;
            double actual = squaredErrorCost.ComputeAverageCost(output, target);

            Assert.AreEqual(expected, actual, 0.0001, "Invalid squared error cost");
        }

        [TestMethod]
        public void SquaredErrorCost_ReturnsCost()
        {
            CostFunction squaredErrorCost = new SquaredErrorCostFunction();
            double output = 1.2;
            double target = 1;
            double expected = 0.02;
            double actual = squaredErrorCost.ComputeCost(output, target);

            Assert.AreEqual(expected, actual, 0.0001, "Invalid squared error cost");
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void SquaredErrorVectorCost_WhenInvalidVectors_ShouldThrowArgument()
        {
            CostFunction squaredErrorCost = new SquaredErrorCostFunction();
            double[] output = new double[] { 0.2, 0.8, 1.2 };
            double[] target = new double[] { 0, 0.7 };
            double actual = squaredErrorCost.ComputeAverageCost(output, target);
        }
    }
}
