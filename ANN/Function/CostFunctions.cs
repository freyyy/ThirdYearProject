using ANN.Core;
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
            return 0.5 * actual.Zip(target, (a, t) => Math.Pow((a - t), 2)).Sum();
        }

        public static double HalfSquaredError(double actual, double target)
        {
            return 0.5 * Math.Pow((actual - target), 2);
        }

        public static double HalfSquaredErrorL2(Network network, double lambda, double[][] actual, double[][] target)
        {
            if (actual.Length != target.Length)
            {
                throw new ArgumentException("Invalid vectors for squared error cost with L2 regularisation. Make sure they are of the same length.");
            }

            for (int i = 0; i < actual.Length; i++)
            {
                if (actual[i].Length != target[i].Length)
                {
                    throw new ArgumentException("Invalid vectors for squared error cost with L2 regularisation. Make sure they are of the same length.");
                }
            }

            double averageSquaredError = actual.Zip(target, (a1d, t1d) => a1d.Zip(t1d, (a, t) => Math.Pow(a - t, 2)).Sum()).Average();
            double weightDecay = 0;

            for (int i = 0; i < network.LayerCount; i++)
            {
                for (int j = 0; j < network[i].NeuronCount; j++)
                {
                    for (int k = 0; k < network[i][j].InputCount; k++)
                    {
                        weightDecay += Math.Pow(network[i][j][k], 2);
                    }
                }
            }

            return averageSquaredError + (lambda / 2 * weightDecay);
        }
    }
}
