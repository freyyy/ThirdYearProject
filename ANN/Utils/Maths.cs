using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ANN.Utils
{
    public static class Maths
    {
        public static double KLDivergence(double x, double y)
        {
            if (x == 0 || x == 1 ||y == 0 || y == 1)
            {
                throw new ArgumentException("Neither argument is allowed values 0 or 1");
            }
            return x * Math.Log(x / y) + (1 - x) * Math.Log((1 - x) / (1 - y));
        }

        public static double KLDivergenceDelta(double x, double y)
        {
            if (x == 0 || x == 1 || y == 0 || y == 1)
            {
                throw new ArgumentException("Neither argument is allowed values 0 or 1");
            }
            return -1 * x / y + (1 - x) / (1 - y);
        }
    }
}
