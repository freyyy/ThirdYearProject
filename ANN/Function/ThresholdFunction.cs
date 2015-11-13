namespace ANN.Function
{
    public class ThresholdFunction : ActivationFunction
    {
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
}
