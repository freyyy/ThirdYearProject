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

        public Neuron(int inputCount, ActivationFunction function)
        {
            _weights = new double[inputCount];
            _inputCount = inputCount;
            _function = function;

            Random rnd = new Random();
            for(int i = 0; i < inputCount; i++)
            {
                _weights[i] = (double) rnd.Next(-10, 10) / 100;
            }

            _bias = (double)rnd.Next(-10, 10) / 100;
        }

        public double this[int i]
        {
            get { return _weights[i]; }
            set { _weights[i] = value; }
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
