using ANN.Core;
using ANN.Function;
using ANN.Learning;
using System;

namespace BackpropagationDemo
{
    class Program
    {
        static void Main(string[] args)
        {
            double[][] inputs = { new double[] { 0, 0 }, new double[] { 0, 1 }, new double[] { 1, 0 }, new double[] { 1, 1 } };
            double[] targets = { 0, 1, 1, 0 };
            double error = 1;
            int epoch = 0;

            ActivationFunction f = new SigmoidFunction();
            Layer layer1 = new Layer(2, 2, f);
            Layer layer2 = new Layer(1, 2, f);
            Network network = new Network(new Layer[] { layer1, layer2 });
            LearningStrategy learning = new BackpropagationLearning(network, 0.25);

            while (error > 0.1)
            {
                error = learning.RunEpoch(inputs, targets);
                epoch++;
                Console.WriteLine("Iteration {0} error: {1}", epoch, error);
            }
            Console.WriteLine("Training complete after {0} epochs using the Backpropagation training regime.", epoch);
        }
    }
}
