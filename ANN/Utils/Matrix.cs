using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ANN.Utils
{
    public static class Matrix
    {
        public static double[][][] AddMatrices(double[][][] xs, double[][][] ys)
        {
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

            double[][][] result = xs.Select(x2d => x2d.Select(x1d => x1d.ToArray()).ToArray()).ToArray();

            for (int i = 0; i < result.Length; i++)
            {
               for (int j = 0; j < result[i].Length; j++)
               {
                   for (int k = 0; k < result[i][j].Length; k++)
                   {
                       result[i][j][k] += ys[i][j][k];
                   }
               }
           }

            //return xs.Zip(ys, (x2d, y2d) => x2d.Zip(y2d, (x1d, y1d) => x1d.Zip(y1d, (x, y) => x + y).ToArray()).ToArray()).ToArray();
            return result;
        }

        public static double[][] AddMatrices(double[][] xs, double[][] ys)
        {
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

            double[][] result = xs.Select(x1d => x1d.ToArray()).ToArray();

            for (int i = 0; i < result.Length; i++)
            {
                for (int j = 0; j < result[i].Length; j++)
                {
                    result[i][j] += ys[i][j];
                }
            }

            // return xs.Zip(ys, (x1d, y1d) => x1d.Zip(y1d, (x, y) => x + y).ToArray()).ToArray();
            return result;
        }
    }
}
