using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ThirdYearProject
{
    public class Layer
    {
        public Neuron[] Neurons { set; get; }

        public int InputCount;

        public int NeuronCount;

        public double[] Output;

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

        public double[] Update(double[] input)
        {
            double[] output = new double[NeuronCount];

            for (int i = 0; i < NeuronCount; i++)
            {
                output[i] = Neurons[i].Update(input);
            }

            return Output = output;
        }
    }
}
