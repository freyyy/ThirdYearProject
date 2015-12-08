namespace ANN.Function
{
    public interface CostFunction
    {
        double ComputeCost(double actual, double target);
        double ComputeAverageCost(double[] actual, double[] target);
    }
}
