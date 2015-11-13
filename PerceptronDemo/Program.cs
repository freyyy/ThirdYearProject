using ANN.Core;
using ANN.Function;
using ANN.Learning;
using log4net;
using log4net.Config;
using System.Configuration;
using System.IO;

namespace PerceptronDemo
{
    class Program
    {
        private static readonly ILog logger = LogManager.GetLogger(typeof(Program));

        static void Main(string[] args)
        {
            double[][] inputs = { new double[] { 0, 0 }, new double[] { 0, 1 }, new double[] { 1, 0 }, new double[] { 1, 1 } };
            double[] targets = { 0, 1, 1, 1 };
            double error = 1;
            int epoch = 0;

            ActivationFunction function = new ThresholdFunction();
            Neuron neuron = new Neuron(2, function);
            LearningStrategy learning = new PerceptronLearning(neuron, 0.25);

            while(error > 0)
            {
                error = learning.RunEpoch(inputs, targets);
                epoch++;
                logger.InfoFormat("Iteration {0} error: {1}", epoch, error);
            }
            logger.InfoFormat("Training complete after {0} epochs using the Perceptron learning regime.", epoch);
        }
    }
}
