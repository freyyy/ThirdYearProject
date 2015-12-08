using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ANN.Core;
using ANN.Function;
using ANN.Learning;

namespace ANN.Test.Learning
{
    [TestClass]
    public class SparseAutoencoderLearningTest
    {
        [TestMethod]
        public void SparseAutoencoderOutputLayerDeltas_ReturnsDeltas()
        {
            ActivationFunction sigmoidFunction = new SigmoidFunction();
            Layer[] layers = new Layer[] { new Layer(2, 2, sigmoidFunction), new Layer(1, 2, sigmoidFunction) };
            Network network = new Network(layers);
            SparseAutoencoderLearning sparseAutoencoder = new SparseAutoencoderLearning(network);
            double[] input = new double[] { 0.5, 0.6 };
            double[] target = new double[] { 1 };
            double[] expected = new double[] { -0.0875168367272141 };

            layers[0][0][0] = 0.1;
            layers[0][0][1] = 0.2;
            layers[0][1][0] = 0.3;
            layers[0][1][1] = 0.4;
            layers[1][0][0] = 0.5;
            layers[1][0][1] = 0.5;
            layers[0][0].Threshold = 0.01;
            layers[0][1].Threshold = 0.02;
            layers[1][0].Threshold = 0.05;

            network.Update(input);
            double[] actual = sparseAutoencoder.OutputLayerDeltas(target);
            Assert.AreEqual(expected[0], actual[0], 0.0001, "Invalid output layer delta");
        }

        [TestMethod]
        public void SparseAutoencoderComputeDeltas_ReturnsDeltas()
        {
            ActivationFunction sigmoidFunction = new SigmoidFunction();
            Layer[] layers = new Layer[] { new Layer(2, 2, sigmoidFunction), new Layer(1, 2, sigmoidFunction) };
            Network network = new Network(layers);
            SparseAutoencoderLearning sparseAutoencoder = new SparseAutoencoderLearning(network);
            double[] input = new double[] { 0.5, 0.6 };
            double[] target = new double[] { 1 };
            double[][] expected = new double[][] { new double[] { -0.0108698887658827, -0.0105735765912387 },
                new double[] { -0.0875168367272141 } };

            layers[0][0][0] = 0.1;
            layers[0][0][1] = 0.2;
            layers[0][1][0] = 0.3;
            layers[0][1][1] = 0.4;
            layers[1][0][0] = 0.5;
            layers[1][0][1] = 0.5;
            layers[0][0].Threshold = 0.01;
            layers[0][1].Threshold = 0.02;
            layers[1][0].Threshold = 0.05;

            network.Update(input);
            double[][] actual = sparseAutoencoder.ComputeDeltas(target);
            Assert.AreEqual(expected[0][0], actual[0][0], 0.0001, "Invalid deltas");
            Assert.AreEqual(expected[0][1], actual[0][1], 0.0001, "Invalid deltas");
            Assert.AreEqual(expected[1][0], actual[1][0], 0.0001, "Invalid deltas");
        }

        [TestMethod]
        public void SparseAutoencoderPartialDerivatives_ReturnsDerivatives()
        {
            ActivationFunction sigmoidFunction = new SigmoidFunction();
            Layer[] layers = new Layer[] { new Layer(2, 2, sigmoidFunction), new Layer(1, 2, sigmoidFunction) };
            Network network = new Network(layers);
            SparseAutoencoderLearning sparseAutoencoder = new SparseAutoencoderLearning(network);
            double[] input = new double[] { 0.5, 0.6 };
            double[] target = new double[] { 1 };
            double[][][] expected = new double[][][]
            {
                new double[][]
                {
                    new double[] { -0.0054349443829414, -0.0065219332595296 },
                    new double[] { -0.0052867882956194, -0.0063441459547432 }
                },
                new double[][]
                {
                    new double[] { -0.0472516427982801, -0.0517626188463657 }
                }
            };


            layers[0][0][0] = 0.1;
            layers[0][0][1] = 0.2;
            layers[0][1][0] = 0.3;
            layers[0][1][1] = 0.4;
            layers[1][0][0] = 0.5;
            layers[1][0][1] = 0.5;
            layers[0][0].Threshold = 0.01;
            layers[0][1].Threshold = 0.02;
            layers[1][0].Threshold = 0.05;

            network.Update(input);
            double[][] deltas = sparseAutoencoder.ComputeDeltas(target);
            double[][][] partialDerivatives = sparseAutoencoder.ComputePartialDerivatives(deltas, input);

            Assert.AreEqual(expected[0][0][0], partialDerivatives[0][0][0], 0.0001, "Invalid partial derivative");
            Assert.AreEqual(expected[0][0][1], partialDerivatives[0][0][1], 0.0001, "Invalid partial derivative");
            Assert.AreEqual(expected[0][1][0], partialDerivatives[0][1][0], 0.0001, "Invalid partial derivative");
            Assert.AreEqual(expected[0][1][1], partialDerivatives[0][1][1], 0.0001, "Invalid partial derivative");
            Assert.AreEqual(expected[1][0][0], partialDerivatives[1][0][0], 0.0001, "Invalid partial derivative");
            Assert.AreEqual(expected[1][0][1], partialDerivatives[1][0][1], 0.0001, "Invalid partial derivative");
        }

        [TestMethod]
        public void SparseAutoencoder_GradientChecking()
        {
            ActivationFunction sigmoidFunction = new SigmoidFunction();
            Layer[] layers = new Layer[] { new Layer(2, 2, sigmoidFunction), new Layer(1, 2, sigmoidFunction) };
            Network network = new Network(layers);
            SparseAutoencoderLearning sparseAutoencoder = new SparseAutoencoderLearning(network);
            double[] input = new double[] { 0.5, 0.6 };
            double[] target = new double[] { 1 };
            double[][][] gradient = new double[][][]
            {
                new double[][]
                {
                    new double[] { 0, 0 },
                    new double[] { 0, 0 }
                },
                new double[][]
                {
                    new double[] { 0, 0 }
                }
            };

            layers[0][0][0] = 0.1;
            layers[0][0][1] = 0.2;
            layers[0][1][0] = 0.3;
            layers[0][1][1] = 0.4;
            layers[1][0][0] = 0.5;
            layers[1][0][1] = 0.5;
            layers[0][0].Threshold = 0.01;
            layers[0][1].Threshold = 0.02;
            layers[1][0].Threshold = 0.05;

            for (int i = 0; i < network.LayerCount; i++)
            {
                for (int j = 0; j < network[i].NeuronCount; j++)
                {
                    for (int k = 0; k < network[i][j].InputCount; k++)
                    {
                        double tmp = network[i][j][k];
                        network[i][j][k] = tmp + 0.0001;
                        gradient[i][j][k] += Math.Pow(network.Update(input)[0] - target[0], 2);
                        network[i][j][k] = tmp - 0.0001;
                        gradient[i][j][k] -= Math.Pow(network.Update(input)[0] - target[0], 2);
                        network[i][j][k] = tmp;
                        gradient[i][j][k]  = 0.5 * gradient[i][j][k] / 0.0002;
                    }
                }
            }

            network.Update(input);
            double[][] deltas = sparseAutoencoder.ComputeDeltas(target);
            double[][][] partialDerivatives = sparseAutoencoder.ComputePartialDerivatives(deltas, input);

            Assert.AreEqual(gradient[0][0][0], partialDerivatives[0][0][0], 0.0001, "Gradient checking failed");
            Assert.AreEqual(gradient[0][0][1], partialDerivatives[0][0][1], 0.0001, "Gradient checking failed");
            Assert.AreEqual(gradient[0][1][0], partialDerivatives[0][1][0], 0.0001, "Gradient checking failed");
            Assert.AreEqual(gradient[0][1][1], partialDerivatives[0][1][1], 0.0001, "Gradient checking failed");
            Assert.AreEqual(gradient[1][0][0], partialDerivatives[1][0][0], 0.0001, "Gradient checking failed");
            Assert.AreEqual(gradient[1][0][1], partialDerivatives[1][0][1], 0.0001, "Gradient checking failed");
        }
    }
}
