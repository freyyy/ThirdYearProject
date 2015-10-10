using System;

namespace ThirdYearProject
{
    public static class Application
    {
        public static void Main()
        {
            double[][] inputs = { new double[] { 0 }, new double[] { 1 } };
            double[] targets = { 0, 1 };
            double error = 100;
            double errorDelta = 100;

            ActivationFunction f = new SigmoidFunction();
            Neuron neuron = new Neuron(1, f);
            LearningStrategy strategy = new DeltaRuleLearning(neuron, 0.1);

            while(error > 0.1)
            {
                double oldError = error;
                error = strategy.RunEpoch(inputs, targets) / 2;
            }
            Console.WriteLine(error);
            Console.WriteLine(neuron.Weights[0]);
            Console.WriteLine(neuron.Threshold);
        }
    } 
}
