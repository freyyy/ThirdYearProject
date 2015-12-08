using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ANN.Utils
{
    public class Matrix
    {
        public static double[][][] AddMatrices(double[][][] xs, double[][][] ys)
        {
            double[][][] result = xs.Select(x => x.ToArray()).ToArray();

            if (xs.Length != ys.Length)
            {
                throw new ArgumentException("Matrices must have same dimensions.");
            }

            for (int i = 0; i < xs.Length; i++)
            {
                if (xs[i].Length != ys[i].Length)
                {
                    throw new ArgumentException("Matrices must have same dimensions.");
                }
            }

            for (int i = 0; i < xs.Length; i++)
            {
                for (int j = 0; j < xs[i].Length; j++)
                {
                    if (xs[i][j].Length != ys[i][j].Length)
                    {
                        throw new ArgumentException("Matrices must have same dimensions.");
                    }
                }
            }

            for (int i = 0; i < xs.Length; i++)
            {
                for (int j = 0; j < xs[i].Length; j++)
                {
                    for (int k = 0; k < xs[i][j].Length; k++)
                    {
                        result[i][j][k] += ys[i][j][k];
                    }
                }
            }

            return result;
        }
    }
}
