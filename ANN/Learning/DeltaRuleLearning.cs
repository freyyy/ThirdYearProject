using ANN.Core;
using System;

namespace ANN.Learning
{
    public class DeltaRuleLearning : LearningStrategy
    {
        private Neuron Neuron;
        private double LearningRate;

        public DeltaRuleLearning(Neuron neuron, double learningRate)
        {
            Neuron = neuron;
            LearningRate = learningRate;
        }

        public double Run(double[] input, double target)
        {
            double output = Neuron.Update(input);
            double error = target - output;
            double deriv = Neuron.Function.OutputDerivative(output);

            for (int i = 0; i < Neuron.InputCount; i++)
            {
                Neuron[i] += LearningRate * deriv * error * input[i];
            }
            Neuron.Bias += LearningRate * deriv * error;

            return Math.Pow(target - output, 2);
        }

        public double RunEpoch(double[][] input, double[] target)
        {
            double error = 0;

            for (int i = 0; i < input.Length; i++)
            {
                error += Run(input[i], target[i]);
            }

            return error / 2;
        }
    }
}
