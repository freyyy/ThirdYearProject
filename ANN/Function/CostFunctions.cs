using System;
using System.Linq;

namespace ANN.Function
{
    public static class CostFunctions
    {
        public static double HalfSquaredError(double[] actual, double[] target)
        {
            if (actual.Length != target.Length)
            {
                throw new ArgumentException("Invalid vectors for squared error cost. Make sure they are of the same length.");
            }
            return 0.5 * actual.Zip(target, (a, t) => Math.Pow((a - t), 2)).Average();
        }

        public static double HalfSquaredError(double actual, double target)
        {
            return 0.5 * Math.Pow((actual - target), 2);
        }
    }
}
