namespace ANN.Learning
{
    public interface LearningStrategy
    {
        double Run(double[] inputs, double target);

        double RunEpoch(double[][] inputs, double[] targets);
    }
}
