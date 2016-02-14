using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlglibTestBench
{
    public class Program
    {
        public static void Function1Grad(double[] x, ref double func, double[] grad, object obj)
        {
            // this callback calculates f(x0,x1) = 100*(x0+3)^4 + (x1-3)^4
            // and its derivatives df/d0 and df/dx1
            func = 100 * Math.Pow(x[0] + 3, 4) + Math.Pow(x[1] - 3, 4);
            Console.WriteLine("Current function value: {0}", func);
            grad[0] = 400 * Math.Pow(x[0] + 3, 3);
            grad[1] = 4 * Math.Pow(x[1] - 3, 3);
        }

        public static void Main(string[] args)
        {
            //
            // This example demonstrates minimization of f(x,y) = 100*(x+3)^4+(y-3)^4
            // using LBFGS method.
            //
            double[] x = new double[] { 0, 0 };
            double epsg = 0.0000000001;
            double epsf = 0;
            double epsx = 0;
            int maxits = 0;
            alglib.minlbfgsstate state;
            alglib.minlbfgsreport rep;

            alglib.minlbfgscreate(1, x, out state);
            alglib.minlbfgssetcond(state, epsg, epsf, epsx, maxits);
            alglib.minlbfgsoptimize(state, Function1Grad, null, null);
            alglib.minlbfgsresults(state, out x, out rep);

            Console.WriteLine("{0}", rep.terminationtype); // EXPECTED: 4
            Console.WriteLine("{0}", alglib.ap.format(x, 2)); // EXPECTED: [-3,3]
            Console.ReadLine();
        }
    }
}
