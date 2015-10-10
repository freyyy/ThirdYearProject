using System;

namespace ThirdYearProject
{
    public interface ActivationFunction {
        double Output(double input);
        double Derivative(double input);
        double OutputDerivative(double input);
    }

    public class ThresholdFunction : ActivationFunction {
        public ThresholdFunction()
        {
        }

        public double Output(double input)
        {
            return input >= 0 ? 1 : 0;
        }

        public double Derivative(double input)
        {
            return 1;
        }

        public double OutputDerivative(double input)
        {
            return 1;
        }
    }

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