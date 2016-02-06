using ANN.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ANN.Utils
{
    public static class Networks
    {
        public static double WeightsSum(Network network)
        {
            double result = 0;

            for (int i = 0; i < network.LayerCount; i++)
            {
                for (int j = 0; j < network[i].NeuronCount; j++)
                {
                    for (int k = 0; k < network[i][j].InputCount; k++)
                    {
                        result += network[i][j][k];
                    }
                }
            }

            return result;
        }

        public static double[][] AverageActivations(Network network, double[][][] activations)
        {
            double[][] average = new double[network.LayerCount][];

            for (int i = 0; i < network.LayerCount; i++)
            {
                average[i] = new double[network[i].NeuronCount];
            }

            if (activations.Length == 0)
            {
                return average;
            }

            for (int i = 0; i < activations.Length; i++)
            {
                for (int j = 0; j < network.LayerCount; j++)
                {
                    for (int k = 0; k < network[j].NeuronCount; k++)
                    {
                        average[j][k] += activations[i][j][k] / activations.Length;
                    }
                }
            }

            return average;
        }
    }
}
