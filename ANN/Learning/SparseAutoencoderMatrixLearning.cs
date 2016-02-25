using ANN.Core;
using ANN.Function;
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
        private Matrix<double> hidden, output;
        private Vector<double> averageHidden, kl;

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

        private Matrix<double> ComputeSigmoidFunction(Matrix<double> x)
        {
            return x.Map(y => 1 / (1 + Math.Exp(-y)));
        }

        public void ComputeKLDelta()
        {
            kl = averageHidden.Map(y => Maths.KLDivergenceDelta(_sparsity, y));
        }

        public void ComputeAverages()
        {
            averageHidden = hidden.RowSums() / hidden.ColumnCount;
        }

        private Matrix<double> ExpandColumn(Vector<double> x, int c)
        {
            Vector<double>[] xMatrix = new Vector<double>[c];

            for (int i = 0; i < c; i++)
            {
                xMatrix[i] = Vector<double>.Build.DenseOfVector(x);
            }

            return M.DenseOfColumnVectors(xMatrix);
        }

        public void ComputeActivations(Matrix<double> x)
        {
            Matrix<double> z1, z2;

            z1 = w1 * x + ExpandColumn(b1, x.ColumnCount);
            hidden = ComputeSigmoidFunction(z1);
            z2 = w2 * hidden + ExpandColumn(b2, x.ColumnCount);
            output = ComputeSigmoidFunction(z2);
        }

        public void ComputePartialDerivatives(Matrix<double> input, Matrix<double> target)
        {
            int batchSize = input.ColumnCount;
            Matrix<double> delta1, delta2, nablaB1, nablaB2;

            delta2 = -(target - output).PointwiseMultiply(output).PointwiseMultiply(1 - output);
            delta1 = (w2.Transpose() * delta2 + _beta * ExpandColumn(kl, input.ColumnCount)).PointwiseMultiply(hidden).PointwiseMultiply(1 - hidden);

            gradW2 = delta2 * hidden.Transpose();
            nablaB2 = delta2;
            gradW1 = delta1 * input.Transpose();
            nablaB1 = delta1;

            gradW2 = gradW2 / batchSize + _lambda * w2;
            gradW1 = gradW1 / batchSize + _lambda * w1;
            gradB2 = nablaB2.RowSums() / batchSize;
            gradB1 = nablaB1.RowSums() / batchSize;
        }
    }
}
