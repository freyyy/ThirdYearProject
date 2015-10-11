using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ThirdYearProject
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

            for (int i = 0; i < _layers.Length; i++)
            {
                output = _layers[i].Update(output);
            }

            return _output = output;
        }
    }
}
