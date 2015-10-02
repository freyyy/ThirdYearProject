using System;
using System.Diagnostics;
using System.Linq;

namespace TLU
{
    public abstract class ThresholdLogicUnit
    {
        protected double LearningRate { set; get; }
        protected double[] Weights { set; get; }
        protected double Threshold { set; get; }

        public ThresholdLogicUnit(int inputCount, double lr = 0.25, double t = 0.5)
        {
            Weights = new double[inputCount];
            Random rnd = new Random();
            for(int i = 0; i < inputCount; i++)
            {
                Weights[i] = (double) rnd.Next(0, 10) / 10;
            }
            LearningRate = lr;
            Threshold = t;
        }

        public int Evaluate(int[] inputs)
        {
            if (inputs.Length != Weights.Length)
                throw new ArgumentException("Invalid number of inputs.");
            return DotProduct(Weights, inputs) >= Threshold ? 1:0;
        }

        private double DotProduct(double[] weights, int[] inputs)
        {
            return weights.Zip(inputs, (w, i) => w * i).Sum();
        }

        public abstract int Learn(int target, int[] inputs);

        public void PrintWeights()
        {
            System.Console.Write("Current weights: ");
            foreach(double weight in Weights)
            {
                System.Console.Write(weight + " ");
            }
            System.Console.WriteLine();
            System.Console.WriteLine("Current threshold: " + Threshold);
        }
    }

    public class PerceptronTLU : ThresholdLogicUnit
    {
        public PerceptronTLU(int inputCount, double lr = 0.25, double t = 0.5) : base(inputCount, lr, t)
        {
        }

        public override int Learn(int target, int[] inputs)
        {
            int output = Evaluate(inputs);
            if (output != target)
            {
                int error = target - output;
                for (int i = 0; i < Weights.Length; i++)
                {
                    Weights[i] += LearningRate * error * inputs[i];
                }
                Threshold += LearningRate * error * (-1);
            }
            return output;
        }
    }

    public static class Run
    {
        public static void Main()
        {
            ThresholdLogicUnit tlu = new PerceptronTLU(2, 0.1);
            int errorCount = 1;
            int[][] inputs = { new int[] { 0, 0 }, new int[] { 0, 1 }, new int[] { 0, 1 }, new int[] { 1, 0 },
                               new int[] { 1, 1 }, new int[] { 0, 0 }, new int[] { 0, 1 }, new int[] { 1, 1 },
                               new int[] { 0, 0 }, new int[] { 1, 0 }};
            int[] targets = { 0, 1, 1, 1, 1, 0, 1, 1, 0, 1};

            while (errorCount != 0)
            {
                errorCount = 0;
                for(int i = 0; i < 10; i++)
                {
                    int output = tlu.Learn(targets[i], inputs[i]);
                    if(output != targets[i])
                    {
                        errorCount++;
                    }
                }
            }

            Debug.Assert(tlu.Evaluate(new int[] { 0, 0 }) == 0);
            Debug.Assert(tlu.Evaluate(new int[] { 0, 1 }) == 1);
            Debug.Assert(tlu.Evaluate(new int[] { 1, 0 }) == 1);
            Debug.Assert(tlu.Evaluate(new int[] { 1, 1 }) == 1);

            System.Console.WriteLine("Perceptron training and verification completed successfully.");
            System.Console.WriteLine("The final perceptron parameters are:");
            tlu.PrintWeights();
        }
    }
}
