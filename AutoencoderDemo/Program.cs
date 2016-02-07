using ANN.Core;
using ANN.Function;
using ANN.Learning;
using System;

namespace AutoencoderDemo
{
    class Program
    {
        static void Main(string[] args)
        {
            double[][] input = { new double[] { 0, 0, 1, 0 }, new double[] { 0, 1, 0, 0 }, new double[] { 1, 0, 0, 0 }, new double[] { 0, 0, 0, 1 } };
            double error = 1;
            int epoch = 0;

            ActivationFunction f = new SigmoidFunction();
            Layer layer1 = new Layer(2, 4, f);
            Layer layer2 = new Layer(4, 2, f);
            Network network = new Network(new Layer[] { layer1, layer2 });
            SparseAutoencoderLearning sae = new SparseAutoencoderLearning(network, 4, 0.5, 0.0001, 0.2, 3);

            while (epoch < 10000)
            {
                error = sae.RunEpoch(input);
                epoch++;
                Console.WriteLine("Error after {0} epochs: {1}", epoch, error);
            }
            Console.WriteLine("Training complete after {0} epochs using the Sparse Autoencoder training regime.", epoch);
            Console.WriteLine("Testing");
            network.Update(new double[] { 1, 0, 0, 0 });
            Console.WriteLine("{0} {1} {2} {3} {4} {5}", network[0][0].Output, network[0][1].Output, network[1][0].Output, network[1][1].Output, network[1][2].Output, network[1][3].Output);
            network.Update(new double[] { 0, 1, 0, 0 });
            Console.WriteLine("{0} {1} {2} {3} {4} {5}", network[0][0].Output, network[0][1].Output, network[1][0].Output, network[1][1].Output, network[1][2].Output, network[1][3].Output);
            network.Update(new double[] { 0, 0, 1, 0 });
            Console.WriteLine("{0} {1} {2} {3} {4} {5}", network[0][0].Output, network[0][1].Output, network[1][0].Output, network[1][1].Output, network[1][2].Output, network[1][3].Output);
            network.Update(new double[] { 0, 0, 0, 1 });
            Console.WriteLine("{0} {1} {2} {3} {4} {5}", network[0][0].Output, network[0][1].Output, network[1][0].Output, network[1][1].Output, network[1][2].Output, network[1][3].Output);

            //List<int[]> trainingData = new List<int[]>();
            //List<int> labels = new List<int>();

            //using (CsvReader csv =
            //       new CsvReader(new StringReader(Properties.Resources.kaggle_train), true))
            //{
            //    int fieldCount = csv.FieldCount;
            //    Console.WriteLine("Field count: {0}", fieldCount);

            //    string[] headers = csv.GetFieldHeaders();

            //    while(csv.ReadNextRecord())
            //    {
            //        List<int> trainingExample = new List<int>();

            //        labels.Add(int.Parse(csv[0]));
            //        for(int i = 1; i < fieldCount; i++)
            //        {
            //            trainingExample.Add(int.Parse(csv[i]));
            //        }

            //        trainingData.Add(trainingExample.ToArray());
            //    }
            //}

            //Console.WriteLine("Total examples: {0}", trainingData.Count);
            //Console.WriteLine("Total labels:   {0}", labels.Count);
        }
    }
}
