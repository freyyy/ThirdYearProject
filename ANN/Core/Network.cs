namespace ANN.Core
{
    public class Network
    {
        private Layer[] _layers;

        private double[] _output;

        private int _layerCount;

        public Network(Layer[] layers)
        {
            _layers = layers;
            _layerCount = layers.Length;
        }

        public Layer this[int i]
        {
            get { return _layers[i]; }
        }

        public int LayerCount
        {
            get { return _layerCount; }
        }

        public double[] Output
        {
            get { return _output; }
        }

        public double[] Update(double[] input)
        {
            double[] output = input;

            for (int i = 0; i < _layerCount; i++)
            {
                output = _layers[i].Update(output);
            }

            _output = output;

            return output;
        }

        public double[][] ComputeNeuronOutputs(double[] input)
        {
            double[][] output = new double[_layerCount][];

            output[0] = _layers[0].Update(input);

            for (int i = 1; i < _layerCount; i++)
            {
                output[i] = _layers[i].Update(output[i - 1]);
            }

            return output;
        }
    }
}
