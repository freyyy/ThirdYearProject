using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ANN.Core;
using ANN.Function;
using ANN.Learning;

namespace DeltaRuleDemo
{
    class Program
    {
        static void Main(string[] args)
        {
            double[][] inputs = { new double[] { 0, 0 }, new double[] { 0, 1 }, new double[] { 1, 0 }, new double[] { 1, 1 } };
            double[] targets = { 0, 1, 1, 1 };
            double error = 1;
            int epoch = 0;

            ActivationFunction function = new SigmoidFunction();
            Neuron neuron = new Neuron(2, function);
            LearningStrategy learning = new DeltaRuleLearning(neuron, 0.25);

            while (error > 0.1)
            {
                error = learning.RunEpoch(inputs, targets);
                epoch++;
                Console.WriteLine("Iteration {0} error: {1}", epoch, error);
            }
            Console.WriteLine("Training complete after {0} epochs using the Delta Rule learning regime.", epoch);
        }
    }
}
