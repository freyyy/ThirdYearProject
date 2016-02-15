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

        public void FunctionValueAndGradient(double[] x, ref double func, double[] grad, object obj)
        {

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
        }
    }
}
