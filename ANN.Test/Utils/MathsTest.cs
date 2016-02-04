using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ANN.Utils;

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
    }
}
