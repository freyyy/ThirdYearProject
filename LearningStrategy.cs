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
}
