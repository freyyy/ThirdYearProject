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
        private Vector<double> averageHidden, deltaKL, kl;

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

        public void ComputeDeltaKL()
        {
            deltaKL = averageHidden.Map(y => Maths.KLDivergenceDelta(_sparsity, y));
        }

        public void ComputeKL()
        {
            kl = averageHidden.Map(y => Maths.KLDivergence(_sparsity, y));
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
            delta1 = (w2.Transpose() * delta2 + _beta * ExpandColumn(deltaKL, input.ColumnCount)).PointwiseMultiply(hidden).PointwiseMultiply(1 - hidden);

            gradW2 = delta2 * hidden.Transpose();
            nablaB2 = delta2;
            gradW1 = delta1 * input.Transpose();
            nablaB1 = delta1;

            gradW2 = gradW2 / batchSize + _lambda * w2;
            gradW1 = gradW1 / batchSize + _lambda * w1;
            gradB2 = nablaB2.RowSums() / batchSize;
            gradB1 = nablaB1.RowSums() / batchSize;
        }

        public double ComputeCost(Matrix<double> target)
        {
            double result = ((output - target).PointwisePower(2).ColumnSums() / 2).Average();
            result += _lambda / 2 * (w1.PointwisePower(2).ColumnSums().Sum() + w2.PointwisePower(2).ColumnSums().Sum());
            result += _beta * (kl.Sum());

            return result;
        }

        public void UpdateWeights(double[] rawParameters)
        {
            int index = 0;

            for (int i = 0; i < w1.RowCount; i++)
            {
                b1[i] = rawParameters[index++];

                for (int j = 0; j < w1.ColumnCount; j++)
                {
                    w1[i, j] = rawParameters[index++];
                }
            }

            for (int i = 0; i < w2.RowCount; i++)
            {
                b2[i] = rawParameters[index++];

                for (int j = 0; j < w2.ColumnCount; j++)
                {
                    w2[i, j] = rawParameters[index++];
                }
            }
        }

        public double[] GradientArray()
        {
            List<double> gradient = new List<double>();

            for (int i = 0; i < gradW1.RowCount; i++)
            {
                gradient.Add(gradB1[i]);

                for (int j = 0; j < gradW1.ColumnCount; j++)
                {
                    gradient.Add(gradW1[i, j]);
                }
            }

            for (int i = 0; i < gradW2.RowCount; i++)
            {
                gradient.Add(gradB2[i]);

                for (int j = 0; j < gradW2.ColumnCount; j++)
                {
                    gradient.Add(gradW2[i, j]);
                }
            }

            return gradient.ToArray();
        }

        public void UpdateUnderlyingNetwork()
        {
            for (int i = 0; i < w1.RowCount; i++)
            {
                _network[0][i].Bias = b1[i];

                for (int j = 0; j < w1.ColumnCount; j++)
                {
                    _network[0][i][j] = w1[i, j];
                }
            }

            for (int i = 0; i < w2.RowCount; i++)
            {
                _network[1][i].Bias = b2[i];

                for (int j = 0; j < w2.ColumnCount; j++)
                {
                    _network[1][i][j] = w2[i, j];
                }
            }
        }

        public void CheckGradient(Matrix<double> input, Matrix<double> target)
        {
            double tmp, gradient;

            for (int i = 0; i < w1.RowCount; i++)
            {
                for (int j = 0; j < w1.ColumnCount; j++)
                {
                    tmp = w1[i, j];
                    w1[i, j] = tmp + 0.0001;

                    ComputeActivations(input);
                    ComputeAverages();
                    ComputeKL();

                    gradient = ComputeCost(target);

                    w1[i, j] = tmp - 0.0001;

                    ComputeActivations(input);
                    ComputeAverages();
                    ComputeKL();

                    gradient -= ComputeCost(target);
                    gradient /= 0.0002;

                    w1[i, j] = tmp;

                    if (Math.Abs(gradW1[i, j] - gradient) > 0.0001)
                    {
                        Console.WriteLine("Gradient checking failed. Expected {0} got {1}. Layer 1 Unit {2} Input {3}",
                            gradW1[i, j], gradient, i, j);
                    }
                    else
                    {
                        Console.WriteLine("Gradient checking passed. Expected {0} got {1}. Layer 1 Unit {2} Input {3}",
                            gradW1[i, j], gradient, i, j);
                    }
                }
            }

            for (int i = 0; i < b1.Count; i++)
            {
                tmp = b1[i];
                b1[i] = tmp + 0.0001;

                ComputeActivations(input);
                ComputeAverages();
                ComputeKL();

                gradient = ComputeCost(target);

                b1[i] = tmp - 0.0001;

                ComputeActivations(input);
                ComputeAverages();
                ComputeKL();

                gradient -= ComputeCost(target);
                gradient /= 0.0002;

                b1[i] = tmp;

                if (Math.Abs(gradB1[i] - gradient) > 0.0001)
                {
                    Console.WriteLine("Gradient checking failed. Expected {0} got {1}. Layer 1 Unit {2} Input Bias",
                        gradB1[i], gradient, i);
                }
                else
                {
                    Console.WriteLine("Gradient checking failed. Expected {0} got {1}. Layer 1 Unit {2} Input Bias",
                        gradB1[i], gradient, i);
                }
            }

            for (int i = 0; i < w2.RowCount; i++)
            {
                for (int j = 0; j < w2.ColumnCount; j++)
                {
                    tmp = w2[i, j];
                    w2[i, j] = tmp + 0.0001;

                    ComputeActivations(input);
                    ComputeAverages();
                    ComputeKL();

                    gradient = ComputeCost(target);

                    w2[i, j] = tmp - 0.0001;

                    ComputeActivations(input);
                    ComputeAverages();
                    ComputeKL();

                    gradient -= ComputeCost(target);
                    gradient /= 0.0002;

                    w2[i, j] = tmp;

                    if (Math.Abs(gradW2[i, j] - gradient) > 0.0001)
                    {
                        Console.WriteLine("Gradient checking failed. Expected {0} got {1}. Layer 2 Unit {2} Input {3}",
                            gradW2[i, j], gradient, i, j);
                    }
                    else
                    {
                        Console.WriteLine("Gradient checking passed. Expected {0} got {1}. Layer 2 Unit {2} Input {3}",
                            gradW2[i, j], gradient, i, j);
                    }
                }
            }

            for (int i = 0; i < b2.Count; i++)
            {
                tmp = b2[i];
                b2[i] = tmp + 0.0001;

                ComputeActivations(input);
                ComputeAverages();
                ComputeKL();

                gradient = ComputeCost(target);

                b2[i] = tmp - 0.0001;

                ComputeActivations(input);
                ComputeAverages();
                ComputeKL();

                gradient -= ComputeCost(target);
                gradient /= 0.0002;

                b2[i] = tmp;

                if (Math.Abs(gradB2[i] - gradient) > 0.0001)
                {
                    Console.WriteLine("Gradient checking failed. Expected {0} got {1}. Layer 2 Unit {2} Input Bias",
                        gradB2[i], gradient, i);
                }
                else
                {
                    Console.WriteLine("Gradient checking failed. Expected {0} got {1}. Layer 2 Unit {2} Input Bias",
                        gradB2[i], gradient, i);
                }
            }
        }

        public Network Network
        {
            get { return _network; }
        }
    }
}
