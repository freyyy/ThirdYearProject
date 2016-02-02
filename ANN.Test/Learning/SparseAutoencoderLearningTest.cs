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
        public void SparseAutoencoderConstructor_InitialisesCache()
        {
            ActivationFunction sigmoidFunction = new SigmoidFunction();
            Layer[] layers = new Layer[] { new Layer(4, 5, sigmoidFunction), new Layer(3, 4, sigmoidFunction) };
            Network network = new Network(layers);
            SparseAutoencoderLearning sparseAutoencoder = new SparseAutoencoderLearning(network);

            int batchSize = sparseAutoencoder.BatchSize;
            double[][][] cachedActivations = sparseAutoencoder.CachedActivations;

            Assert.AreEqual(1, batchSize, 0, "Inalid batch size");
            Assert.AreEqual(batchSize, cachedActivations.Length, 0, "Invalid activations cache size");
            
            for (int i = 0; i < batchSize; i++)
            {
                Assert.AreEqual(network.LayerCount, cachedActivations[i].Length);

                for (int j = 0; j < network.LayerCount; j++)
                {
                    Assert.AreEqual(network[j].NeuronCount, cachedActivations[i][j].Length);
                }
            }

            sparseAutoencoder = new SparseAutoencoderLearning(network, 32);

            batchSize = sparseAutoencoder.BatchSize;
            cachedActivations = sparseAutoencoder.CachedActivations;

            Assert.AreEqual(32, batchSize, 0, "Inalid batch size");
            Assert.AreEqual(batchSize, cachedActivations.Length, 0, "Invalid activations cache size");

            for (int i = 0; i < batchSize; i++)
            {
                Assert.AreEqual(network.LayerCount, cachedActivations[i].Length);

                for (int j = 0; j < network.LayerCount; j++)
                {
                    Assert.AreEqual(network[j].NeuronCount, cachedActivations[i][j].Length);
                }
            }
        }

        [TestMethod]
        public void SparseAutoencoderUpdateCachedActivations_UpdatesCache()
        {
            ActivationFunction sigmoidFunction = new SigmoidFunction();
            Layer[] layers = new Layer[] { new Layer(2, 2, sigmoidFunction), new Layer(1, 2, sigmoidFunction) };
            Network network = new Network(layers);
            SparseAutoencoderLearning sparseAutoencoder = new SparseAutoencoderLearning(network);
            double[][] input = new double[][] { new double[] { 0.5, 0.6 } };
            double[][][] expected = new double[][][]
            {
                new double[][]
                {
                    new double[] { 0.539914884556, 0.591458978433 },
                    new double[] { 0.626138674824 }
                },
                new double[][]
                {
                    new double[] { 0.547357618143, 0.608259030747 },
                    new double[] { 0.628971793540 }
                }
            };

            layers[0][0][0] = 0.1;
            layers[0][0][1] = 0.2;
            layers[0][1][0] = 0.3;
            layers[0][1][1] = 0.4;
            layers[1][0][0] = 0.5;
            layers[1][0][1] = 0.5;
            layers[0][0].Bias = 0.01;
            layers[0][1].Bias = 0.02;
            layers[1][0].Bias = 0.05;

            double[][][] cachedActivations = sparseAutoencoder.UpdateCachedActivations(input);

            Assert.AreEqual(expected[0][0][0], cachedActivations[0][0][0], 0.0001, "Invalid cached activation value");
            Assert.AreEqual(expected[0][0][1], cachedActivations[0][0][1], 0.0001, "Invalid cached activation value");
            Assert.AreEqual(expected[0][1][0], cachedActivations[0][1][0], 0.0001, "Invalid cached activation value");

            input = new double[][] { new double[] { 0.5, 0.6 }, new double[] { 0.6, 0.7 } };
            sparseAutoencoder = new SparseAutoencoderLearning(network, 2);

            cachedActivations = sparseAutoencoder.UpdateCachedActivations(input);

            Assert.AreEqual(expected[0][0][0], cachedActivations[0][0][0], 0.0001, "Invalid cached activation value");
            Assert.AreEqual(expected[0][0][1], cachedActivations[0][0][1], 0.0001, "Invalid cached activation value");
            Assert.AreEqual(expected[0][1][0], cachedActivations[0][1][0], 0.0001, "Invalid cached activation value");
            Assert.AreEqual(expected[1][0][0], cachedActivations[1][0][0], 0.0001, "Invalid cached activation value");
            Assert.AreEqual(expected[1][0][1], cachedActivations[1][0][1], 0.0001, "Invalid cached activation value");
            Assert.AreEqual(expected[1][1][0], cachedActivations[1][1][0], 0.0001, "Invalid cached activation value");
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void SparseAutoencoderUpdateCachedActivations_ThrowsArgument()
        {
            ActivationFunction sigmoidFunction = new SigmoidFunction();
            Layer[] layers = new Layer[] { new Layer(2, 2, sigmoidFunction), new Layer(1, 2, sigmoidFunction) };
            Network network = new Network(layers);
            SparseAutoencoderLearning sparseAutoencoder = new SparseAutoencoderLearning(network, 2);
            double[][] input = new double[][] { new double[] { 0.5, 0.6 } };

            layers[0][0][0] = 0.1;
            layers[0][0][1] = 0.2;
            layers[0][1][0] = 0.3;
            layers[0][1][1] = 0.4;
            layers[1][0][0] = 0.5;
            layers[1][0][1] = 0.5;
            layers[0][0].Bias = 0.01;
            layers[0][1].Bias = 0.02;
            layers[1][0].Bias = 0.05;

            double[][][] cachedActivations = sparseAutoencoder.UpdateCachedActivations(input);
        }

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
            layers[0][0].Bias = 0.01;
            layers[0][1].Bias = 0.02;
            layers[1][0].Bias = 0.05;

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
            double[][] input = new double[][] { new double[] { 0.5, 0.6 } };
            double[][] target = new double[][] { new double[] { 1 } };
            double[][] expected = new double[][]
            {
                new double[] { -0.0108698887658827, -0.0105735765912387 },
                new double[] { -0.0875168367272141 }
            };

            layers[0][0][0] = 0.1;
            layers[0][0][1] = 0.2;
            layers[0][1][0] = 0.3;
            layers[0][1][1] = 0.4;
            layers[1][0][0] = 0.5;
            layers[1][0][1] = 0.5;
            layers[0][0].Bias = 0.01;
            layers[0][1].Bias = 0.02;
            layers[1][0].Bias = 0.05;

            sparseAutoencoder.UpdateCachedActivations(input);
            double[][] actual = sparseAutoencoder.ComputeDeltas(0, target[0]);
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
            double[][] input = new double[][] { new double[] { 0.5, 0.6 } };
            double[][] target = new double[][] { new double[] { 1 } };
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
            layers[0][0].Bias = 0.01;
            layers[0][1].Bias = 0.02;
            layers[1][0].Bias = 0.05;

            sparseAutoencoder.UpdateCachedActivations(input);
            double[][] deltas = sparseAutoencoder.ComputeDeltas(0, target[0]);
            double[][][] partialDerivatives = sparseAutoencoder.ComputePartialDerivatives(0, deltas, input[0]);

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
            double[][] input = new double[][] { new double[] { 0.5, 0.6 } };
            double[][] target = new double[][] { new double[] { 1 } };
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
            layers[0][0].Bias = 0.01;
            layers[0][1].Bias = 0.02;
            layers[1][0].Bias = 0.05;

            for (int i = 0; i < network.LayerCount; i++)
            {
                for (int j = 0; j < network[i].NeuronCount; j++)
                {
                    for (int k = 0; k < network[i][j].InputCount; k++)
                    {
                        double tmp = network[i][j][k];
                        network[i][j][k] = tmp + 0.0001;
                        gradient[i][j][k] += Math.Pow(network.Update(input[0])[0] - target[0][0], 2);
                        network[i][j][k] = tmp - 0.0001;
                        gradient[i][j][k] -= Math.Pow(network.Update(input[0])[0] - target[0][0], 2);
                        network[i][j][k] = tmp;
                        gradient[i][j][k]  = 0.5 * gradient[i][j][k] / 0.0002;
                    }
                }
            }

            sparseAutoencoder.UpdateCachedActivations(input);
            double[][] deltas = sparseAutoencoder.ComputeDeltas(0, target[0]);
            double[][][] partialDerivatives = sparseAutoencoder.ComputePartialDerivatives(0, deltas, input[0]);

            Assert.AreEqual(gradient[0][0][0], partialDerivatives[0][0][0], 0.0001, "Gradient checking failed");
            Assert.AreEqual(gradient[0][0][1], partialDerivatives[0][0][1], 0.0001, "Gradient checking failed");
            Assert.AreEqual(gradient[0][1][0], partialDerivatives[0][1][0], 0.0001, "Gradient checking failed");
            Assert.AreEqual(gradient[0][1][1], partialDerivatives[0][1][1], 0.0001, "Gradient checking failed");
            Assert.AreEqual(gradient[1][0][0], partialDerivatives[1][0][0], 0.0001, "Gradient checking failed");
            Assert.AreEqual(gradient[1][0][1], partialDerivatives[1][0][1], 0.0001, "Gradient checking failed");
        }

        [TestMethod]
        public void SparseAutoencoder_BatchGradientChecking()
        {
            ActivationFunction sigmoidFunction = new SigmoidFunction();
            Layer[] layers = new Layer[] { new Layer(2, 2, sigmoidFunction), new Layer(1, 2, sigmoidFunction) };
            Network network = new Network(layers);
            SparseAutoencoderLearning sparseAutoencoder = new SparseAutoencoderLearning(network, 3);
            double[][] input = new double[][] { new double[] { 0.5, 0.6 }, new double[] { 0.1, 0.2 }, new double[] { 0.3, 0.3 } };
            double[][] target = new double[][] { new double[] { 1 }, new double[] { 0 }, new double[] { 0.5 } };
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
            layers[0][0].Bias = 0.01;
            layers[0][1].Bias = 0.02;
            layers[1][0].Bias = 0.05;

            for (int i = 0; i < network.LayerCount; i++)
            {
                for (int j = 0; j < network[i].NeuronCount; j++)
                {
                    for (int k = 0; k < network[i][j].InputCount; k++)
                    {
                        double tmp = network[i][j][k];
                        network[i][j][k] = tmp + 0.0001;

                        for (int l = 0; l < input.Length; l++)
                        {
                            gradient[i][j][k] += ((double)1 / input.Length) * 0.5 * Math.Pow(network.Update(input[l])[0] - target[l][0], 2);
                        }

                        network[i][j][k] = tmp - 0.0001;

                        for (int l = 0; l < input.Length; l++)
                        {
                            gradient[i][j][k] -= ((double)1 / input.Length) * 0.5 * Math.Pow(network.Update(input[l])[0] - target[l][0], 2);
                        }

                        network[i][j][k] = tmp;
                        gradient[i][j][k] = gradient[i][j][k] / 0.0002;
                    }
                }
            }

            double[][][] partialDerivatives = sparseAutoencoder.ComputeBatchPartialDerivatives(input, target);

            Assert.AreEqual(gradient[0][0][0], ((double)1 / input.Length) * partialDerivatives[0][0][0], 0.0001, "Gradient checking failed");
            Assert.AreEqual(gradient[0][0][1], ((double)1 / input.Length) * partialDerivatives[0][0][1], 0.0001, "Gradient checking failed");
            Assert.AreEqual(gradient[0][1][0], ((double)1 / input.Length) * partialDerivatives[0][1][0], 0.0001, "Gradient checking failed");
            Assert.AreEqual(gradient[0][1][1], ((double)1 / input.Length) * partialDerivatives[0][1][1], 0.0001, "Gradient checking failed");
            Assert.AreEqual(gradient[1][0][0], ((double)1 / input.Length) * partialDerivatives[1][0][0], 0.0001, "Gradient checking failed");
            Assert.AreEqual(gradient[1][0][1], ((double)1 / input.Length) * partialDerivatives[1][0][1], 0.0001, "Gradient checking failed");
        }
    }
}
