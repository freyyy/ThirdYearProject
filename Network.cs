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
    }
}
