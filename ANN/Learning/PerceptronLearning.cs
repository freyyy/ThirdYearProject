using System;
using ANN.Core;

namespace ANN.Learning
{
    public class PerceptronLearning : LearningStrategy
    {
        private Neuron Neuron;
        private double LearningRate;

        public PerceptronLearning(Neuron neuron, double learningRate)
        {
            Neuron = neuron;
            LearningRate = learningRate;
        }

        public double Run(double[] input, double target)
        {
            double output = Neuron.Update(input);
            double error = target - output;

            for (int i = 0; i < Neuron.InputCount; i++)
            {
                Neuron[i] += LearningRate * error * input[i];
            }
            Neuron.Bias += LearningRate * error;

            return Math.Abs(error);
        }

        public double RunEpoch(double[][] input, double[] target)
        {
            double error = 0;

            for (int i = 0; i < input.Length; i++)
            {
                error += Run(input[i], target[i]);
            }

            return error;
        }
    }
}
