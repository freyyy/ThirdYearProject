using System;

namespace ANN.Function
{
    public class SigmoidFunction : ActivationFunction
    {
        private double Rho;

        public SigmoidFunction()
        {
            Rho = 1.0;
        }

        public SigmoidFunction(double rho)
        {
            Rho = rho;
        }

        public double Output(double input)
        {
            return 1 / (1 + Math.Exp(-input));
        }

        public double Derivative(double input)
        {
            double output = Output(input);
            return output * (1 - output);
        }

        public double OutputDerivative(double input)
        {
            return input * (1 - input);
        }
    }
}
