using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ThirdYearProject
{
    class Network
    {
        public Layer[] Layers { set; get; }

        public double[] Output;

        public Network(Layer[] layers)
        {
            Layers = layers;
        }

        public double[] Update(double[] input)
        {
            double[] output = input;

            for (int i = 0; i < Layers.Length; i++)
            {
                output = Layers[i].Update(output);
            }

            return Output = output;
        }
    }
}
