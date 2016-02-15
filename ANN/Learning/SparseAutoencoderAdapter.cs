using ANN.Core;
using ANN.Function;
using ANN.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ANN.Learning
{
    public class SparseAutoencoderAdapter
    {
        private SparseAutoencoderLearning _sparseAutoencoder;
        private double[][] _input;
        private double _lastValue = double.MaxValue;
        private double _iterations = 0;

        public SparseAutoencoderAdapter(SparseAutoencoderLearning sparseAutoencoder, double[][] input)
        {
            _sparseAutoencoder = sparseAutoencoder;
            _input = input;
        }

        private void UpdateFunctionParameters(double[] x)
        {
            int index = 0;
            Network network = _sparseAutoencoder.Network;

            for (int i = 0; i < network.LayerCount; i++)
            {
                for (int j = 0; j < network[i].NeuronCount; j++)
                {
                    network[i][j].Bias = x[index++];

                    for (int k = 0; k < network[i][j].InputCount; k++)
                    {
                        network[i][j][k] = x[index++];
                    }
                }
            }
        }

        private void ComputeFunctionGradients(double[] grad)
        {
            int index = 0;
            Tuple<double[][][], double[][]> functionGradients = _sparseAutoencoder.ComputeBatchPartialDerivatives(_input, _input);
            
            for (int i = 0; i < functionGradients.Item1.Length; i++)
            {
                for (int j = 0; j < functionGradients.Item1[i].Length; j++)
                {
                    grad[index++] = functionGradients.Item2[i][j];

                    for (int k = 0; k < functionGradients.Item1[i][j].Length; k++)
                    {
                        grad[index++] = functionGradients.Item1[i][j][k];
                    }
                }
            }
        }

        private void ComputeFunctionValue(ref double func)
        {
            double[][] averageActivations = Networks.AverageActivations(_sparseAutoencoder.Network, _sparseAutoencoder.CachedActivations);
            double[][] output = new double[_sparseAutoencoder.BatchSize][];

            for (int i = 0; i < _sparseAutoencoder.BatchSize; i++)
            {
                output[i] = new double[_sparseAutoencoder.Network[_sparseAutoencoder.Network.LayerCount - 1].NeuronCount];

                for (int j = 0; j < output[i].Length; j++)
                {
                    output[i][j] = _sparseAutoencoder.CachedActivations[i][_sparseAutoencoder.Network.LayerCount - 1][j];
                }
            }

            func = CostFunctions.HalfSquaredErrorL2Sparsity(_sparseAutoencoder.Network, averageActivations[0], _sparseAutoencoder.Sparsity, _sparseAutoencoder.Lambda, _sparseAutoencoder.Beta, output, _input);
        }

        public void FunctionValueAndGradient(double[] x, ref double func, double[] grad, object obj)
        {
            UpdateFunctionParameters(x);
            ComputeFunctionGradients(grad);
            ComputeFunctionValue(ref func);
        }

        public void PrintProgress()
        {

        }

        public SparseAutoencoderLearning SparseAutoencoderLearning
        {
            get { return _sparseAutoencoder; }
        }

        public double[][] Input
        {
            get { return _input; }
            set { _input = value; }
        }
    }
}
