using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ANN.Function
{
    public class SquaredErrorCostFunction : CostFunction
    {
        public SquaredErrorCostFunction()
        {
        }

        public double ComputeAverageCost(double[] actual, double[] target)
        {
            if (actual.Length != target.Length)
            {
                throw new ArgumentException("Invalid vectors for squared error cost. Make sure they are of the same length.");
            }
            return 0.5 * actual.Zip(target, (a, t) => Math.Pow((a - t), 2)).Average();
        }

        public double ComputeCost(double actual, double target)
        {
            return 0.5 * Math.Pow((actual - target), 2);
        }
    }
}
