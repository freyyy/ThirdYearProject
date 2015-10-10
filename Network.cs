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

        public Network(Layer[] layers)
        {
            Layers = layers;
        }

        public double[] Output(double[] input)
        {
            double[] output = input;

            for(int i = 0; i < Layers.Length; i++)
            {
                output = Layers[i].Output(output);
            }

            return output;
        }
    }
}
