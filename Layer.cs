using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ThirdYearProject
{
    class Layer
    {
        public Neuron[] Neurons { set; get; }

        public int InputCount;

        public int NeuronCount;

        public Layer(int neuronCount, int inputCount, ActivationFunction function)
        {
            InputCount = inputCount;
            NeuronCount = neuronCount;
            Neurons = new Neuron[neuronCount];
            for(int i = 0; i < neuronCount; i++)
            {
                Neurons[i] = new Neuron(inputCount, function);
            }
        }

        public double[] Output(double[] inputs)
        {
            double[] output = new double[NeuronCount];

            for(int i = 0; i < NeuronCount; i++)
            {
                output[i] = Neurons[i].Output(inputs);
            }

            return output;
        }
    }
}
