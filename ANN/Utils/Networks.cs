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
    }
}
