using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ANN.Utils;

namespace ANN.Test.Utils
{
    [TestClass]
    public class VectorTest
    {
        [TestMethod]
        public void VectorDotProduct_WithValidVectors_ReturnsResult()
        {
            double[] vector1 = new double[] { 1, 2, 3 };
            double[] vector2 = new double[] { 1, 2, 3 };

            double expected = 14;
            double actual = Vectors.DotProduct(vector1, vector2);

            Assert.AreEqual(expected, actual, 0.0001, "Incorrect dot product");
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void VectorDotProduct_WhenVectorsSizeNotEqual_ShouldThrowArgument()
        {
            double[] vector1 = new double[] { 1, 2, 3 };
            double[] vector2 = new double[] { 1, 2 };

            double actual = Vectors.DotProduct(vector1, vector2);
        }
    }
}
