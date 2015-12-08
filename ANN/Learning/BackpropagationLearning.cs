using System;
using ANN.Core;
using ANN.Function;

namespace ANN.Learning
{
    public class BackpropagationLearning : LearningStrategy
    {
        private Network _network;

        public double LearningRate;

        public BackpropagationLearning(Network network, double learningRate)
        {
            _network = network;
            LearningRate = learningRate;
        }

        private void PrintDeltas(double[][] delta)
        {
            for (int i = 0; i < delta.Length; i++)
            {
                for (int j = 0; j < delta[i].Length; j++)
                {
                    Console.Write(delta[i][j] + " ");
                }
                Console.WriteLine();
            }
        }

        private double[][] ApproximatePartialDerivatives(Network network, double[] target)
        {
            double[][] derivatives = new double[network.LayerCount][];

            return new double[][] { new double[] { 0, 0 }, new double[] { 0, 0 } };
        }

        public double Run(double[] input, double target)
        {
            // Feed-forward - compute network output
            _network.Update(input);

            int layerCount = _network.LayerCount;
            double error = 0;

            double[][] delta = new double[layerCount][];

            for (int i = 0; i < layerCount; i++)
            {
                delta[i] = new double[_network[i].NeuronCount];
            }

            // Compute output layer deltas
            for (int i = 0; i < _network[layerCount - 1].NeuronCount; i++)
            {
                Neuron currentNeuron = _network[layerCount - 1][i];
                error += Math.Abs(target - currentNeuron.Output);
                delta[layerCount - 1][i] = (target - currentNeuron.Output) *
                    currentNeuron.Function.OutputDerivative(currentNeuron.Output);
            }

            // Compute hidden layers deltas
            for (int i = layerCount - 2; i >= 0; i--)
            {
                for (int j = 0; j < _network[i].NeuronCount; j++)
                {
                    Neuron currentNeuron = _network[i][j];
                    for (int k = 0; k < _network[i + 1].NeuronCount; k++)
                    {
                        delta[i][j] += delta[i + 1][k] * _network[i + 1][k][j];
                    }
                    delta[i][j] *= currentNeuron.Function.OutputDerivative(currentNeuron.Output);
                }
            }

            // Update weights and thresholds
            for (int i = 0; i < layerCount; i++)
            {
                for (int j = 0; j < _network[i].NeuronCount; j++)
                {
                    for (int k = 0; k < _network[i].InputCount; k++)
                    {
                        _network[i][j][k] += LearningRate * delta[i][j] * input[k];
                    }
                    _network[i][j].Threshold += LearningRate * delta[i][j] * (-1);
                }
                input = _network[i].Output;
            }

            return error;
        }

        public double RunEpoch(double[][] inputs, double[] targets)
        {
            double error = 0;

            for (int i = 0; i < inputs.Length; i++)
            {
                error += Run(inputs[i], targets[i]);
            }

            return error;
        }
    }
}
