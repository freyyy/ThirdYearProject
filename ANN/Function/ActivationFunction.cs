namespace ANN.Function
{
    public interface ActivationFunction
    {
        double Output(double input);
        double Derivative(double input);
        double OutputDerivative(double input);
    }
}