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

            return xs.Zip(ys, (x2d, y2d) => x2d.Zip(y2d, (x1d, y1d) => x1d.Zip(y1d, (x, y) => x + y).ToArray()).ToArray()).ToArray();
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

            return xs.Zip(ys, (x1d, y1d) => x1d.Zip(y1d, (x, y) => x + y).ToArray()).ToArray();
        }
    }
}
