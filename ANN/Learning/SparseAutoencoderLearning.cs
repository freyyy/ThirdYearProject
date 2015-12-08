using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ANN.Core;

namespace ANN.Learning
{
    public class SparseAutoencoderLearning
    {
        private Network _network;

        public SparseAutoencoderLearning(Network network)
        {
            _network = network;
        }

        public double[] OutputLayerDeltas(double[] target)
        {
            double[] outputLayerActivations = _network[_network.LayerCount - 1].Output;
            double[] deltas = target.Zip(outputLayerActivations, (t, a) => (-1) * (t - a) * a * (1 - a)).ToArray();

            return deltas;
        }

        public double[][] ComputeDeltas(double[] target)
        {
            int layerCount = _network.LayerCount;
            double[][] deltas = new double[layerCount][];
            double[] output = _network[layerCount - 1].Output;

            if (target.Length != _network[layerCount - 1].NeuronCount)
            {
                throw new ArgumentException("Target vector size must be the same as network output size.");
            }

            for (int i = 0; i < layerCount; i++)
            {
                deltas[i] = new double[_network[i].NeuronCount];
            }

            deltas[layerCount - 1] = target.Zip(output, (t, o) => (-1) * (t - o) * o * (1 - o)).ToArray();

            for (int i = layerCount - 2; i >= 0; i--)
            {
                double[] prevDeltas = deltas[i + 1];
                double[] weights = new double[prevDeltas.Length];

                for (int j = 0; j < _network[i].NeuronCount; j++)
                {
                    double neuronOutput = _network[i][j].Output;

                    for (int k = 0; k < weights.Length; k++)
                    {
                        weights[k] = _network[i + 1][k][j];
                    }
                    deltas[i][j] = weights.Zip(prevDeltas, (w, d) => w * d).Sum() * neuronOutput * (1 - neuronOutput);
                }
            }

            return deltas;
        }

        public double[][][] ComputePartialDerivatives(double[][] deltas, double[] input)
        {
            int layerCount = Network.LayerCount;
            int neuronCount, inputCount;
            double[][][] partialDerivatives = new double[layerCount][][];

            for (int i = 0; i < layerCount; i++)
            {
                neuronCount = _network[i].NeuronCount;
                partialDerivatives[i] = new double[neuronCount][];

                for (int j = 0; j < neuronCount; j++)
                {
                    inputCount = _network[i][j].InputCount;
                    partialDerivatives[i][j] = new double[inputCount];
                }
            }

            for (int i = 0; i < layerCount; i++)
            {
                neuronCount = _network[i].NeuronCount;

                for (int j = 0; j < neuronCount; j++)
                {
                    inputCount = _network[i][j].InputCount;

                    for(int k = 0; k < inputCount; k++)
                    {
                        partialDerivatives[i][j][k] = deltas[i][j] * input[k];
                    }
                }
                input = _network[i].Output;
            }

            return partialDerivatives;
        }

        public Network Network
        {
            get { return _network; }
        }
    }
}
