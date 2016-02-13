using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ANN.Utils;
using System.Linq;

namespace ANN.Test.Utils
{
    [TestClass]
    public class MathsTest
    {
        [TestMethod]
        public void KLDivergence_ReturnsValue()
        {
            double x = 0.5, y = 0.5;

            double actual = Maths.KLDivergence(x, y);

            Assert.AreEqual(0, actual, 0.0001, "Invalid KL divergence value.");

            x = 0.4;
            y = 0.2;

            actual = Maths.KLDivergence(x, y);

            Assert.AreEqual(0.4 * Math.Log(2) + 0.6 * Math.Log(0.75), actual, 0.0001, "Invalid KL divergence value.");

            x = 0.7;
            y = 0.1;

            actual = Maths.KLDivergence(x, y);

            Assert.AreEqual(0.7 * Math.Log(7) + 0.3 * Math.Log(0.3 / 0.9), actual, 0.0001, "Invalid KL divergence value.");
        }

        [TestMethod]
        public void KLDivergenceDelta_ReturnsValue()
        {
            double x = 0.5, y = 0.5;

            double actual = Maths.KLDivergenceDelta(x, y);

            Assert.AreEqual(0, actual, 0.0001, "Invalid KL divergence value.");

            x = 0.4;
            y = 0.2;

            actual = Maths.KLDivergenceDelta(x, y);

            Assert.AreEqual(-1.25, actual, 0.0001, "Invalid KL divergence value.");

            x = 0.7;
            y = 0.1;

            actual = Maths.KLDivergenceDelta(x, y);

            Assert.AreEqual(-6.66666666, actual, 0.0001, "Invalid KL divergence value.");
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void KLDivergenceFirst_ThrowsArgument()
        {
            double x = 0, y = 5;

            double actual = Maths.KLDivergence(x, y);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void KLDivergenceSecond_ThrowsArgument()
        {
            double x = 1, y = 5;

            double actual = Maths.KLDivergence(x, y);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void KLDivergenceThird_ThrowsArgument()
        {
            double x = 5, y = 0;

            double actual = Maths.KLDivergence(x, y);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void KLDivergenceFourth_ThrowsArgument()
        {
            double x = 5, y = 1;

            double actual = Maths.KLDivergence(x, y);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void KLDivergenceDeltaFirst_ThrowsArgument()
        {
            double x = 0, y = 5;

            double actual = Maths.KLDivergenceDelta(x, y);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void KLDivergenceDeltaSecond_ThrowsArgument()
        {
            double x = 1, y = 5;

            double actual = Maths.KLDivergenceDelta(x, y);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void KLDivergenceDeltaThird_ThrowsArgument()
        {
            double x = 5, y = 0;

            double actual = Maths.KLDivergenceDelta(x, y);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void KLDivergenceDeltaFourth_ThrowsArgument()
        {
            double x = 5, y = 1;

            double actual = Maths.KLDivergenceDelta(x, y);
        }

        [TestMethod]
        public void StandardDeviation_ReturnsStandardDeviation()
        {
            double[] input = new double[] { 5 };
            double actual = Maths.StandardDeviation(input);

            Assert.AreEqual(0, actual, 0.0001, "Invalid standard deviation");

            input = new double[] { };
            actual = Maths.StandardDeviation(input);

            Assert.AreEqual(0, actual, 0.0001, "Invalid standard deviation");

            input = new double[] { -0.5, -0.1, 0.3, 0.7, 2.1, 3};
            actual = Maths.StandardDeviation(input);

            Assert.AreEqual(1.23884, actual, 0.0001, "Invalid standard deviation");
        }

        [TestMethod]
        public void RemoveDcComponent_RemovesDcComponent()
        {
            double[][] input = new double[][]
            {
                new double[] { 1, 2, 3, 4, 5 },
                new double[] { 3, 4, 5, 6, 7 }
            };
            double[][] actual = Maths.RemoveDcComponent(input);
            double[] mean = input.Select(i => i.Average()).ToArray();

            for (int i = 0; i < actual.Length; i++)
            {
                for (int j = 0; j < actual[i].Length; j++)
                {
                    Assert.AreEqual(input[i][j] - mean[i], actual[i][j], 0.0001, "DC component not removed correctly");
                }
            }
        }

        [TestMethod]
        public void Rescale_RescalesInput()
        {
            double[] input = { -3, -2, 1, 4 };
            double[] actual = Maths.Rescale(input, 0, 1);

            Assert.AreEqual(input.Length, actual.Length, 0, "Incorrect normalised input length");
            Assert.AreEqual(0.0, actual[0], 0.0001, "Incorrect normalised input");
            Assert.AreEqual(1.0 / 7, actual[1], 0.0001, "Incorrect normalised input");
            Assert.AreEqual(4.0 / 7, actual[2], 0.0001, "Incorrect normalised input");
            Assert.AreEqual(1.0, actual[3], 0.0001, "Incorrect normalised input");
        }

        // TODO: Come up with a suitable test
        [TestMethod]
        public void TruncatedAndRescale_TruncatesAndRescalesInput()
        {
            
        }
    }
}
