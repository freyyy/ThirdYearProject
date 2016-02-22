using ANN.Core;
using ANN.Utils;
using System;
using System.Linq;
using System.Threading.Tasks;

namespace ANN.Function
{
    public static class CostFunctions
    {
        public static double HalfSquaredError(double[][] actual, double[][] target)
        {
            if (actual.Length != target.Length)
            {
                throw new ArgumentException("Invalid vectors for half squared error cost. Make sure they are of the same length.");
            }

            for (int i = 0; i < actual.Length; i++)
            {
                if (actual[i].Length != target[i].Length)
                {
                    throw new ArgumentException("Invalid vectors for half squared error cost. Make sure they are of the same length.");
                }
            }

            return actual.Zip(target, (a1d, t1d) => 0.5 * a1d.Zip(t1d, (a, t) => Math.Pow(a - t, 2)).Sum()).Average();
        }

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

            double averageSquaredError = actual.Zip(target, (a1d, t1d) => 0.5 * a1d.Zip(t1d, (a, t) => Math.Pow(a - t, 2)).Sum()).Average();
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

        public static double HalfSquaredErrorL2Sparsity(Network network, double[] averageActivations,
            double sparsity, double lambda, double beta, double[][] actual, double[][] target)
        {
            if (actual.Length != target.Length)
            {
                throw new ArgumentException("Invalid vectors for squared error cost with L2 regularisation and sparsity. Make sure they are of the same length.");
            }

            for (int i = 0; i < actual.Length; i++)
            {
                if (actual[i].Length != target[i].Length)
                {
                    throw new ArgumentException("Invalid vectors for squared error cost with L2 regularisation and sparsity. Make sure they are of the same length.");
                }
            }

            if (averageActivations.Length != network[0].NeuronCount)
            {
                throw new ArgumentException("Invalid size of average activations array.");
            }

            double cost = HalfSquaredErrorL2(network, lambda, actual, target);

            for (int i = 0; i < averageActivations.Length; i++)
            {
                cost += beta * Maths.KLDivergence(sparsity, averageActivations[i]);
            }

            return cost;
        }

        public static double ParallelHalfSquaredErrorL2Sparsity(Network network, double[] averageActivations,
            double sparsity, double lambda, double beta, double[][] actual, double[][] target)
        {
            double cost = 0;
            object lockCost = new object();

            Parallel.For(0, actual.Length,
                () => 0.0d,

                (x, loopState, partialResult) =>
                {
                    for (int i = 0; i < actual[x].Length; i++)
                    {
                        partialResult += 0.5 * Math.Pow((actual[x][i] - target[x][i]), 2);
                    }

                    return partialResult;
                },

                (localPartialCost) =>
                {
                    lock (lockCost)
                    {
                        cost += localPartialCost;
                    }
                }
            );

            cost = cost / actual.Length;

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

            cost += lambda / 2 * weightDecay;

            for (int i = 0; i < averageActivations.Length; i++)
            {
                cost += beta * Maths.KLDivergence(sparsity, averageActivations[i]);
            }

            return cost;
        }
    }
}
