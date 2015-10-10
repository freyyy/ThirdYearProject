using System;

namespace ThirdYearProject
{
    public class Neuron
    {
        public double[] Weights { set; get; }

        public double Threshold { set; get; }

        public double Output;

        public ActivationFunction Function;

        public Neuron(int inputsCount, ActivationFunction function)
        {
            Weights = new double[inputsCount];
            Threshold = 0.5;
            Function = function;

            Random rnd = new Random();
            for(int i = 0; i < inputsCount; i++)
            {
                Weights[i] = (double) rnd.Next(0, 10) / 10;
            }
        }

        public double Activation(double[] inputs)
        {
            return Utils.DotProduct(inputs, Weights) - Threshold;
        }

        public double Update(double[] input)
        {
            return Output = Function.Output(Activation(input));
        }
    }
}
