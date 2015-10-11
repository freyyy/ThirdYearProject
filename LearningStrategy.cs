using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ThirdYearProject
{
    public interface LearningStrategy
    {
        double Run(double[] inputs, double target);

        double RunEpoch(double[][] inputs, double[] targets);
    }

    public class PerceptronLearning : LearningStrategy
    {
        private Neuron Neuron;
        private double LearningRate;

        public PerceptronLearning(Neuron neuron, double learningRate)
        {
            Neuron = neuron;
            LearningRate = learningRate;
        }

        public double Run(double[] inputs, double target)
        {
            double output = Neuron.Update(inputs);
            double error = target - output;

            for(int i = 0; i < Neuron.Weights.Length; i++)
            {
                Neuron.Weights[i] += LearningRate * error * inputs[i];
            }
            Neuron.Threshold += LearningRate * error * (-1);

            return Math.Abs(error);
        }

        public double RunEpoch(double[][] inputs, double[] targets)
        {
            double error = 0;

            for(int i = 0; i < inputs.Length; i++)
            {
                error += Run(inputs[i], targets[i]);
            }

            return error;
        }
    }

    public class DeltaRuleLearning : LearningStrategy
    {
        private Neuron Neuron;
        private double LearningRate;

        public DeltaRuleLearning(Neuron neuron, double learningRate)
        {
            Neuron = neuron;
            LearningRate = learningRate;
        }

        public double Run(double[] inputs, double target)
        {
            double output = Neuron.Update(inputs);
            double error = target - output;
            double deriv = Neuron.Function.Derivative(Neuron.Activation(inputs));

            for(int i = 0; i < Neuron.Weights.Length; i ++)
            {
                Neuron.Weights[i] += LearningRate * deriv * error * inputs[i];
            }
            Neuron.Threshold += LearningRate * deriv * error * (-1);

            return Math.Abs(error);
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

    public class BackpropagationLearning : LearningStrategy
    {
        public Network Network;

        public double LearningRate;

        public BackpropagationLearning(Network network, double learningRate)
        {
            Network = network;
            LearningRate = learningRate;
        }

        private void PrintDeltas(double[][] delta)
        {
            for(int i = 0; i < delta.Length; i++)
            {
                for (int j = 0; j < delta[i].Length; j++)
                {
                    Console.Write(delta[i][j] + " ");
                }
                Console.WriteLine();
            }
        }

        public double Run(double[] input, double target)
        {
            // Feed-forward - compute network output
            Network.Update(input);

            Layer[] layers = Network.Layers;
            int layerCount = layers.Length;
            double error = 0;
            
            double[][] delta = new double[layerCount][];

            for(int i = 0; i < layerCount; i++)
            {
                delta[i] = new double[layers[i].NeuronCount];
            }

            // Compute output layer deltas
            for(int i = 0; i < layers[layerCount - 1].NeuronCount; i++)
            {
                Neuron currentNeuron = layers[layerCount - 1].Neurons[i];
                error += Math.Abs(target - currentNeuron.Output);
                delta[layerCount - 1][i] = (target - currentNeuron.Output) * 
                    currentNeuron.Function.OutputDerivative(currentNeuron.Output);
            }

            // Compute hidden layers deltas
            for(int i = layerCount - 2; i >= 0; i--)
            {
                for(int j = 0; j < layers[i].NeuronCount; j++)
                {
                    Neuron currentNeuron = layers[i].Neurons[j];
                    for(int k = 0; k < layers[i + 1].NeuronCount; k++)
                    {
                        delta[i][j] += delta[i + 1][k] * layers[i + 1].Neurons[k].Weights[i];
                    }
                    delta[i][j] *= currentNeuron.Function.OutputDerivative(currentNeuron.Output);
                }
            }

            // Update weights and thresholds
            for(int i = 0; i < layerCount; i++)
            {
                for(int j = 0; j < layers[i].NeuronCount; j++)
                {
                    for(int k = 0; k < layers[i].InputCount; k++)
                    {
                        layers[i].Neurons[j].Weights[k] += LearningRate * delta[i][j] * input[k];
                    }
                    layers[i].Neurons[j].Threshold += LearningRate * delta[i][j] * (-1);
                }
                input = layers[i].Output;
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
