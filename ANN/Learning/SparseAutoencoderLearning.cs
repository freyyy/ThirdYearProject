using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ANN.Core;
using ANN.Utils;
using ANN.Function;

namespace ANN.Learning
{
    public class SparseAutoencoderLearning
    {
        private Network _network;
        private int _batchSize;
        private double _sparsity;
        private double _beta;
        private double _lambda;
        private double _alpha;
        private double[][][] _cachedActivations;

        public SparseAutoencoderLearning(Network network, double alpha) : this(network, 1, alpha) { }

        public SparseAutoencoderLearning(Network network, int batchSize, double alpha) : this(network, batchSize, alpha, 0) { }

        public SparseAutoencoderLearning(Network network, int batchSize, double alpha, double lambda) : this(network, batchSize, alpha, lambda, 0, 0) { }

        public SparseAutoencoderLearning(Network network, int batchSize, double alpha, double lambda, double sparsity, double beta)
        {
            _network = network;
            _batchSize = batchSize;
            _alpha = alpha;
            _lambda = lambda;
            _sparsity = sparsity;
            _beta = beta;

            _cachedActivations = new double[batchSize][][];

            for (int i = 0; i < batchSize; i++)
            {
                _cachedActivations[i] = new double[network.LayerCount][];

                for (int j = 0; j < network.LayerCount; j++)
                {
                    _cachedActivations[i][j] = new double[network[j].NeuronCount];
                }
            }
        }

        public double[][][] UpdateCachedActivations(double[][] input)
        {
            if (input.Length != _batchSize)
            {
                throw new ArgumentException("Input size must match the batch size exactly.");
            }

            for (int i = 0; i < input.Length; i++)
            {
                _network.Update(input[i]);

                for (int j = 0; j < _network.LayerCount; j++)
                {
                    for (int k = 0; k < _network[j].NeuronCount; k++)
                    {
                        _cachedActivations[i][j][k] = _network[j][k].Output;
                    }
                }
            }

            return _cachedActivations;
        }

        public double[] OutputLayerDeltas(double[] target)
        {
            double[] outputLayerActivations = _network[_network.LayerCount - 1].Output;
            double[] deltas = target.Zip(outputLayerActivations, (t, a) => (-1) * (t - a) * a * (1 - a)).ToArray();

            return deltas;
        }

        public double[][] ComputeDeltas(int batchIndex, double[][] averageActivations, double[] target)
        {
            int layerCount = _network.LayerCount;
            double[][] deltas = new double[layerCount][];
            double[] output = _cachedActivations[batchIndex][layerCount - 1];

            if (target.Length != _network[layerCount - 1].NeuronCount)
            {
                throw new ArgumentException("Target vector size must be the same as network output size.");
            }

            for (int i = 0; i < layerCount; i++)
            {
                deltas[i] = new double[_network[i].NeuronCount];
            }

            deltas[layerCount - 1] = target.Zip(output, (t, o) => (-1) * (t - o) * o * (1 - o)).ToArray();

            for (int i = layerCount - 2; i >= 0; i--)
            {
                double[] prevDeltas = deltas[i + 1];
                double[] weights = new double[prevDeltas.Length];

                for (int j = 0; j < _network[i].NeuronCount; j++)
                {
                    double neuronOutput = _cachedActivations[batchIndex][i][j];

                    for (int k = 0; k < weights.Length; k++)
                    {
                        weights[k] = _network[i + 1][k][j];
                    }
                    deltas[i][j] = (weights.Zip(prevDeltas, (w, d) => w * d).Sum() + _beta * Maths.KLDivergenceDelta(_sparsity, averageActivations[i][j])) * neuronOutput * (1 - neuronOutput);
                }
            }

            return deltas;
        }

        public double[][][] ComputePartialDerivatives(int batchIndex, double[][] deltas, double[] input)
        {
            int layerCount = Network.LayerCount;
            int neuronCount, inputCount;
            double[][][] partialDerivatives = new double[layerCount][][];

            for (int i = 0; i < layerCount; i++)
            {
                neuronCount = _network[i].NeuronCount;
                partialDerivatives[i] = new double[neuronCount][];

                for (int j = 0; j < neuronCount; j++)
                {
                    inputCount = _network[i][j].InputCount;
                    partialDerivatives[i][j] = new double[inputCount];
                }
            }

            for (int i = 0; i < layerCount; i++)
            {
                neuronCount = _network[i].NeuronCount;

                for (int j = 0; j < neuronCount; j++)
                {
                    inputCount = _network[i][j].InputCount;

                    for (int k = 0; k < inputCount; k++)
                    {
                        partialDerivatives[i][j][k] = deltas[i][j] * input[k];
                    }
                }
                input = _cachedActivations[batchIndex][i];
            }

            return partialDerivatives;
        }

        public Tuple<double[][][], double[][]> ComputeBatchPartialDerivatives(double[][] input, double[][] target)
        {
            int layerCount = Network.LayerCount;
            int neuronCount, inputCount;
            double[][] deltas;
            double[][] partialDerivativesBias = new double[layerCount][];
            double[][] averageActivations;
            double[][][] partialDerivativesWeights = new double[layerCount][][];
            double[][][] tmpPartialDerivatives;

            for (int i = 0; i < layerCount; i++)
            {
                neuronCount = _network[i].NeuronCount;
                partialDerivativesWeights[i] = new double[neuronCount][];
                partialDerivativesBias[i] = new double[neuronCount];

                for (int j = 0; j < neuronCount; j++)
                {
                    inputCount = _network[i][j].InputCount;
                    partialDerivativesWeights[i][j] = new double[inputCount];
                }
            }

            UpdateCachedActivations(input);
            averageActivations = Networks.AverageActivations(_network, _cachedActivations);

            for(int i = 0; i< input.Length; i++)
            {
               deltas = ComputeDeltas(i, averageActivations, target[i]);
               tmpPartialDerivatives = ComputePartialDerivatives(i, deltas, input[i]);

               partialDerivativesWeights = Matrix.AddMatrices(partialDerivativesWeights, tmpPartialDerivatives);
               partialDerivativesBias = Matrix.AddMatrices(partialDerivativesBias, deltas);
            }

            for (int i = 0; i < _network.LayerCount; i++)
            {
                for (int j = 0; j < _network[i].NeuronCount; j++)
                {
                    for (int k = 0; k < _network[i][j].InputCount; k++)
                    {
                        partialDerivativesWeights[i][j][k] = (partialDerivativesWeights[i][j][k] / input.Length) +_lambda * _network[i][j][k];
                    }
                    partialDerivativesBias[i][j] = partialDerivativesBias[i][j] / input.Length;
                }
            }

            return Tuple.Create(partialDerivativesWeights, partialDerivativesBias);
        }

        public Network UpdateNetworkParameters(double[][][] weightGradients, double[][] biasGradients)
        {
            for (int i = 0; i < _network.LayerCount; i++)
            {
                for (int j = 0; j < _network[i].NeuronCount; j++)
                {
                    _network[i][j].Bias -= _alpha * biasGradients[i][j];

                    for (int k = 0; k < _network[i][j].InputCount; k++)
                    {
                        _network[i][j][k] -= _alpha * weightGradients[i][j][k];
                    }
                }
            }

            return _network;
        }

        public double RunEpoch(double[][] input)
        {
            double[][] output = new double[_batchSize][];
            int layerCount = _network.LayerCount;

            Tuple<double[][][], double[][]> gradients = ComputeBatchPartialDerivatives(input, input);

            for (int i = 0; i < output.Length; i++)
            {
                output[i] = new double[_network[layerCount - 1].NeuronCount];

                for (int j = 0; j < output[i].Length; j++)
                {
                    output[i][j] = _cachedActivations[i][layerCount - 1][j];
                }
            }

            double[][] averageActivations = Networks.AverageActivations(_network, _cachedActivations);
            double cost = CostFunctions.HalfSquaredErrorL2Sparsity(_network, averageActivations[0], _sparsity, _lambda, _beta, output, input);

            UpdateNetworkParameters(gradients.Item1, gradients.Item2);

            return cost;
        }

        public Network Network
        {
            get { return _network; }
        }

        public int BatchSize
        {
            get { return _batchSize; }
        }

        public double Sparsity
        {
            get { return _sparsity; }
        }

        public double Beta
        {
            get { return _beta; }
        }

        public double Lambda
        {
            get { return _lambda; }
        }

        public double Alpha
        {
            get { return _alpha; }
        }

        public double[][][] CachedActivations
        {
            get { return _cachedActivations;  }
        }
    }
}
