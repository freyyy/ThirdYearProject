using System;
using System.Linq;

namespace ANN.Utils
{
    public static class Vectors
    {
        public static double DotProduct(double[] xs, double[] ys)
        {
            if (xs.Length != ys.Length)
            {
                throw new ArgumentException("Invalid vectors for dot product. Make sure they are of the same length.");
            }
            return xs.Zip(ys, (x, y) => x * y).Sum();
        }
    } 
}