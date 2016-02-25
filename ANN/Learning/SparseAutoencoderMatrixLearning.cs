using ANN.Core;
using ANN.Utils;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ANN.Learning
{
    public class SparseAutoencoderMatrixLearning
    {
        private Network _network;
        private double _lambda;
        private double _sparsity;
        private double _beta;
        private bool _checkGradient;

        MatrixBuilder<double> M = Matrix<double>.Build;
        VectorBuilder<double> V = Vector<double>.Build;

        private Vector<double> b1, b2;
        private Vector<double> gradB1, gradB2;
        private Matrix<double> w1, w2;
        private Matrix<double> gradW1, gradW2;
        private Matrix<double> hidden, output, kl;
        private Matrix<double> averageHidden;

        public SparseAutoencoderMatrixLearning(Network network, double lambda, double sparsity, double beta, bool checkGradient = false)
        {
            _network = network;
            _lambda = lambda;
            _sparsity = sparsity;
            _beta = beta;
            _checkGradient = checkGradient;

            double[][] rows = new double[_network[0].NeuronCount][];
            double[] row = new double[_network[0].NeuronCount];

            for (int i = 0; i < rows.Length; i++)
            {
                rows[i] = _network[0][i].Weights;
                row[i] = _network[0][i].Bias;
            }

            w1 = M.DenseOfRowArrays(rows);
            b1 = V.DenseOfArray(row);

            rows = new double[_network[1].NeuronCount][];
            row = new double[_network[1].NeuronCount];

            for (int i = 0; i < rows.Length; i++)
            {
                rows[i] = _network[1][i].Weights;
                row[i] = _network[1][i].Bias;
            }

            w2 = M.DenseOfRowArrays(rows);
            b2 = V.DenseOfArray(row);
        }

        public void ComputeBatchPartialDerivatives(double[][] averageActivations, double[,] inputArray, double[,] targetArray)
        {
            int batchSize = inputArray.Length;

            Matrix<double> target = Matrix<double>.Build.DenseOfArray(targetArray);
            Matrix<double> input = Matrix<double>.Build.DenseOfArray(inputArray);

            Matrix<double> delta2 = -(target - output).PointwiseMultiply(output).PointwiseMultiply(1 - output);
            Matrix<double> delta1 = (w2.Transpose() * delta2 + _beta * kl).PointwiseMultiply(hidden).PointwiseMultiply(1 - hidden);

            gradW2 = delta2 * hidden.Transpose();
            Matrix<double> nablaB2 = delta2;
            gradW1 = delta1 * input.Transpose();
            Matrix<double> nablaB1 = delta1;

            gradW2 = gradW2 / batchSize + _lambda * w2;
            gradW1 = gradW1 / batchSize + _lambda * w1;
            gradB2 = nablaB2.RowSums() / batchSize;
            gradB1 = nablaB1.RowSums() / batchSize;
        }
    }
}
