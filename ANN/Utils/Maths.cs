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

        public static double StandardDeviation(double[] input)
        {
            double ret = 0;
            int count = input.Count();

            if (count > 1)
            {
                double average = input.Average();
                double sum = input.Sum(i => (i - average) * (i - average));

                ret = Math.Sqrt(sum / count);
            }
            
            return ret;
        }

        public static double[][] RemoveDcComponent(double[][] input)
        {
            double mean;

            for (int i = 0; i < input.Length; i++)
            {
                mean = input[i].Average();
                input[i] = input[i].Select(p => p - mean).ToArray();
            }

            return input;
        }

        public static double[] Rescale(double[] input, double minValue, double maxValue)
        {
            double min = input.Min();
            double max = input.Max();
            return input.Select(i => (i - min) * (maxValue - minValue) / (max - min) + minValue).ToArray();
        }

        public static double[][] Rescale(double[][] input, double minValue, double maxValue)
        {
            double[][] result = new double[input.Length][];

            for (int i = 0; i < result.Length; i++)
            {
                result[i] = Rescale(input[i], minValue, maxValue);
            }

            return result;
        }

        public static double[][] TruncateAndRescale(double[][] input, double minValue, double maxValue)
        {
            double pstd;

            for (int i = 0; i < input.Length; i++)
            {
                pstd = 3 * StandardDeviation(input[i]);
                input[i] = input[i].Select(p => Math.Max(Math.Min(p, pstd), -pstd) / pstd).ToArray();
                input[i] = Rescale(input[i], 0.1, 0.9);
            }

            return input;
        }
    }
}
