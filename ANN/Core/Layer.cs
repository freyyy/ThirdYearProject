using ANN.Function;
using System.Threading.Tasks;

namespace ANN.Core
{
    public class Layer
    {
        private Neuron[] _neurons;

        private int _inputCount;

        private int _neuronCount;

        private double[] _output;

        public Layer(int neuronCount, int inputCount, ActivationFunction function)
        {
            _inputCount = inputCount;
            _neuronCount = neuronCount;

            _neurons = new Neuron[neuronCount];
            for(int i = 0; i < neuronCount; i++)
            {
                _neurons[i] = new Neuron(inputCount, function);
            }
        }

        public Neuron this[int i]
        {
            get { return _neurons[i]; }
        }

        public int InputCount
        {
            get { return _inputCount; }
        }

        public int NeuronCount
        {
            get { return _neuronCount; }
        }

        public double[] Output
        {
            get { return _output; }
        }

        public double[] Update(double[] input)
        {
            double[] output = new double[NeuronCount];

            for (int i = 0; i < NeuronCount; i++)
            {
                output[i] = _neurons[i].Update(input);
            }

            _output = output;

            return output;
        }
    }
}
