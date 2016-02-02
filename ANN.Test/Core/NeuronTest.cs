using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ANN.Core;
using ANN.Function;

namespace ANN.Test.Core
{
    [TestClass]
    public class NeuronTest
    {
        [TestMethod]
        public void NeuronUpdate_UpdatesOutputAndReturnsValue()
        {
            ActivationFunction sigmoidFunction = new SigmoidFunction();
            Neuron sigmoidNeuron = new Neuron(3, sigmoidFunction);

            double[] input = new double[] { 0.3, 0.2, 0.1 };
            double expected = 0.512497396484;

            sigmoidNeuron[0] = 0.1;
            sigmoidNeuron[1] = 0.2;
            sigmoidNeuron[2] = 0.3;
            sigmoidNeuron.Bias = 0.05;

            double actual = sigmoidNeuron.Update(input);
            double neuronOutput = sigmoidNeuron.Output;

            Assert.AreEqual(expected, actual, 0.0001, "Invalid neuron output");
            Assert.AreEqual(expected, neuronOutput, 0.0001, "Invalid neuron output");
        }

        [TestMethod]
        public void NeuronActivation_ReturnsNeuronActivation()
        {
            ActivationFunction sigmoidFunction = new SigmoidFunction();
            Neuron sigmoidNeuron = new Neuron(3, sigmoidFunction);

            double[] input = new double[] { 0.3, 0.2, 0.1 };
            double expected = 0.05;

            sigmoidNeuron[0] = 0.1;
            sigmoidNeuron[1] = 0.2;
            sigmoidNeuron[2] = 0.3;
            sigmoidNeuron.Bias = 0.05;

            double actual = sigmoidNeuron.Activation(input);

            Assert.AreEqual(expected, actual, 0.0001, "Invalid neuron activation");
        }
    }
}
