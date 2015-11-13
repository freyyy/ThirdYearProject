using System;

namespace ThirdYearProject
{
    public static class Application
    {
        public static void Main()
        {
            double[][] inputs = { new double[] { 0, 0 }, new double[] { 0, 1 }, new double[] { 1, 0 }, new double[] { 1, 1 } };
            double[] targets = { 0, 1, 1, 0 };
            double error = 1;

            ActivationFunction f = new SigmoidFunction();
            Layer layer1 = new Layer(2, 2, f);
            Layer layer2 = new Layer(1, 2, f);
            Network network = new Network(new Layer[] { layer1, layer2 });

            layer1[0][0] = 0.2;
            layer1[0][1] = 0.3;
            layer1[1][0] = 0.3;
            layer1[1][1] = 0.4;
            layer2[0][0] = 0.5;
            layer2[0][1] = 0.5;
            layer1[0].Threshold = 0.4;
            layer1[1].Threshold = 0.2;
            layer2[0].Threshold = 0.8;

            LearningStrategy learning = new BackpropagationLearning(network, 0.1);
            
            while(error > 0.1)
            {
                error = learning.RunEpoch(inputs, targets);
                Console.WriteLine(error);
            }
            Console.WriteLine(network.Update(new double[] { 0, 0 })[0]);
            Console.WriteLine(network.Update(new double[] { 0, 1 })[0]);
            Console.WriteLine(network.Update(new double[] { 1, 0 })[0]);
            Console.WriteLine(network.Update(new double[] { 1, 1 })[0]);
        }
    } 
}
