using ANN.Core;
using ANN.Function;
using ANN.Learning;
using ANN.Utils;
using LumenWorks.Framework.IO.Csv;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NumericsTestBench
{
    public class Program
    {
        public static void Main(string[] args)
        {
            // Using managed code only
            Control.UseManaged();
            Console.WriteLine(Control.LinearAlgebraProvider);

            // Initialise two matrices of size 1000x1000 with random numbers.
            var m = Matrix<double>.Build.Random(1000, 1000);
            var v = Matrix<double>.Build.Random(1000, 1000);

            var w = Stopwatch.StartNew();
            var y1 = m * v;
            Console.WriteLine(w.Elapsed);

            // Using the Intel MKL native provider
            Control.UseNativeMKL();
            Console.WriteLine(Control.LinearAlgebraProvider);

            w.Restart();
            var y2 = m * v;
            Console.WriteLine(w.Elapsed);
        }
    }
}
