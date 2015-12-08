using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ANN.Utils;

namespace ANN.Test.Utils
{
    [TestClass]
    public class MatrixTest
    {
        [TestMethod]
        public void MatrixAdd_WithValidMatrices_ReturnsResult()
        {
            double[][][] matrix1 = new double[][][] {
                new double[][] {
                    new double[] { 0, 1 },
                    new double[] { 2, 3, 4 } },
                new double[][] {
                    new double[] { 5 } } };
            double[][][] matrix2 = new double[][][] {
                new double[][] {
                    new double[] { 5, 4 },
                    new double[] { 3, 2, 1 } },
                new double[][] {
                    new double[] { 0 } } };
            double[][][] expected = new double[][][] {
                new double[][] {
                    new double[] { 5, 5 },
                    new double[] { 5, 5, 5 } },
                new double[][] {
                    new double[] { 5 } } };

            double[][][] actual = Matrix.AddMatrices(matrix1, matrix2);

            for (int i = 0; i < actual.Length; i++)
            {
                for (int j = 0; j < actual[i].Length; j++)
                {
                    CollectionAssert.AreEqual(expected[i][j], actual[i][j], "Incorrect matrix addition");
                }
            }
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void MatrixAdd_WithInvalidMatricesFirst_ThrowsArgument()
        {
            double[][][] matrix1 = new double[][][] {
                new double[][] {
                    new double[] { 0, 1 },
                    new double[] { 2, 3, 4 } },
                new double[][] {
                    new double[] { 5 } } };
            double[][][] matrix2 = new double[][][] {
                new double[][] {
                    new double[] { 3, 2, 1} } };
            double[][][] expected = new double[][][] {
                new double[][] {
                    new double[] { 5, 5 },
                    new double[] { 5, 5, 5 } },
                new double[][] {
                    new double[] { 5 } } };

            double[][][] actual = Matrix.AddMatrices(matrix1, matrix2);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void MatrixAdd_WithInvalidMatricesSecond_ThrowsArgument()
        {
            double[][][] matrix1 = new double[][][] {
                new double[][] {
                    new double[] { 0, 1 },
                    new double[] { 2, 3, 4 } },
                new double[][] {
                    new double[] { 5 } } };
            double[][][] matrix2 = new double[][][] {
                new double[][] {
                    new double[] { 3, 2, 1} },
                new double[][] {
                    new double[] { 0 } } };
            double[][][] expected = new double[][][] {
                new double[][] {
                    new double[] { 5, 5 },
                    new double[] { 5, 5, 5 } },
                new double[][] {
                    new double[] { 5 } } };

            double[][][] actual = Matrix.AddMatrices(matrix1, matrix2);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void MatrixAdd_WithInvalidMatricesThird_ThrowsArgument()
        {
            double[][][] matrix1 = new double[][][] {
                new double[][] {
                    new double[] { 0, 1 },
                    new double[] { 2, 3, 4 } },
                new double[][] {
                    new double[] { 5 } } };
            double[][][] matrix2 = new double[][][] {
                new double[][] {
                    new double[] { 5, 4 },
                    new double[] { 3, 2} },
                new double[][] {
                    new double[] { 0 } } };
            double[][][] expected = new double[][][] {
                new double[][] {
                    new double[] { 5, 5 },
                    new double[] { 5, 5, 5 } },
                new double[][] {
                    new double[] { 5 } } };

            double[][][] actual = Matrix.AddMatrices(matrix1, matrix2);
        }
    }
}
