using ANN.Core;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ANN.Learning
{
    public class SparseAutoencoderMatrixAdapter
    {
        private SparseAutoencoderMatrixLearning _sparseAutoencoder;
        private Matrix<double> _input, _target;
        private double[] _parameters;
        private int _iterations = 0;

        public SparseAutoencoderMatrixAdapter(SparseAutoencoderMatrixLearning sparseAutoencoder, Matrix<double> input, Matrix<double> target)
        {
            _sparseAutoencoder = sparseAutoencoder;
            _input = input;
            _target = target;

            List<double> parameters = new List<double>();
            Network network = sparseAutoencoder.Network;

            for (int i = 0; i < network.LayerCount; i++)
            {
                for (int j = 0; j < network[i].NeuronCount; j++)
                {
                    parameters.Add(network[i][j].Bias);

                    for (int k = 0; k < network[i][j].InputCount; k++)
                    {
                        parameters.Add(network[i][j][k]);
                    }
                }
            }

            _parameters = parameters.ToArray();
        }

        private void ComputeGradient(double[] grad)
        {
            _sparseAutoencoder.ComputeActivations(_input);
            _sparseAutoencoder.ComputeAverages();
            _sparseAutoencoder.ComputeDeltaKL();
            _sparseAutoencoder.ComputePartialDerivatives(_input, _target);

            double[] gradient = _sparseAutoencoder.GradientArray();

            for (int i = 0; i < gradient.Length; i++)
            {
                grad[i] = gradient[i];
            }
        }

        private void ComputeValue(ref double func)
        {
            _sparseAutoencoder.ComputeKL();

            func = _sparseAutoencoder.ComputeCost(_target);
        }

        public void FunctionValueAndGradient(double[] x, ref double func, double[] grad, object obj)
        {
            _sparseAutoencoder.UpdateWeights(x);

            ComputeGradient(grad);
            ComputeValue(ref func);
        }

        public void PrintProgress(double[] arg, double func, object obj)
        {
            _iterations++;
            Console.WriteLine("Function value after {0} iterations: {1}", _iterations, func);
        }
    }
}
