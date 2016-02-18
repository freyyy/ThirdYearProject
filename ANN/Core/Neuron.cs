using System;
using ANN.Function;
using ANN.Utils;

namespace ANN.Core
{
    public class Neuron
    {
        private double[] _weights;

        private double _bias;

        private double _output;

        private int _inputCount;

        private ActivationFunction _function;

        private static readonly Random rnd = new Random();

        private static double MIN_WEIGHT = -Math.Sqrt(6) / Math.Sqrt(129);

        private static double MAX_WEIGHT = Math.Sqrt(6) / Math.Sqrt(129);

        public Neuron(int inputCount, ActivationFunction function)
        {
            _weights = new double[inputCount];
            _inputCount = inputCount;
            _function = function;

            for(int i = 0; i < inputCount; i++)
            {
                _weights[i] = GetInitialWeight();
            }

            _bias = 0;
        }

        private double GetInitialWeight()
        {
            return rnd.NextDouble() * (MAX_WEIGHT - MIN_WEIGHT) + MIN_WEIGHT;
        }

        public double this[int i]
        {
            get { return _weights[i]; }
            set { _weights[i] = value; }
        }

        public double[] Weights
        {
            get { return _weights; }
        }

        public double Bias
        {
            get { return _bias; }
            set { _bias = value; }
        }

        public int InputCount
        {
            get { return _inputCount; }
        }

        public double Output
        {
            get { return _output; }
        }

        public ActivationFunction Function
        {
            get { return _function; }
        }

        public double Activation(double[] input)
        {
            return Vector.DotProduct(input, _weights) + _bias;
        }

        public double Update(double[] input)
        {
            return _output = _function.Output(Activation(input));
        }
    }
}
