using ANN.Core;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Xml.Serialization;

namespace ANN.Utils
{
    public static class Networks
    {
        public static Random rng = new Random();

        public static double WeightsSum(Network network)
        {
            double result = 0;

            for (int i = 0; i < network.LayerCount; i++)
            {
                for (int j = 0; j < network[i].NeuronCount; j++)
                {
                    for (int k = 0; k < network[i][j].InputCount; k++)
                    {
                        result += network[i][j][k];
                    }
                }
            }

            return result;
        }

        public static double[][] AverageActivations(Network network, double[][][] activations)
        {
            double[][] average = new double[network.LayerCount][];

            for (int i = 0; i < network.LayerCount; i++)
            {
                average[i] = new double[network[i].NeuronCount];
            }

            if (activations.Length == 0)
            {
                return average;
            }

            for (int i = 0; i < activations.Length; i++)
            {
                for (int j = 0; j < network.LayerCount; j++)
                {
                    for (int k = 0; k < network[j].NeuronCount; k++)
                    {
                        average[j][k] += activations[i][j][k] / activations.Length;
                    }
                }
            }

            return average;
        }

        public static double[] ComputeMaximumActivationInput(double[] weights)
        {
            double norm = Math.Sqrt(weights.Select(w => w * w).Sum());

            return weights.Select(w => w / norm).ToArray();
        }

        public static double[][] ComputeMaximumActivationInput(double[][] weights)
        {
            double[][] result = new double[weights.Length][];

            for (int i = 0; i < result.Length; i++)
            {
                result[i] = ComputeMaximumActivationInput(weights[i]);
            }

            return result;
        }

        public static void ExportParametersToText(Network network)
        {
            XmlSerializer xmlSerialiser = new XmlSerializer(typeof(double[][][]));

            double[][][] parameters = new double[network.LayerCount][][];

            for (int i = 0; i < parameters.Length; i++)
            {
                parameters[i] = new double[network[i].NeuronCount][];

                for (int j = 0; j < parameters[i].Length; j++)
                {
                    parameters[i][j] = network[i][j].Weights;
                }
            }

            using (StreamWriter writer = new StreamWriter("parameters.txt"))
            {
                xmlSerialiser.Serialize(writer, parameters);
            }
        }

        public static void ExportReconstructionsToBitmap(Network network, Matrix<double> data, int dataPerRow, int dataPerColumn, int dataWidth, int dataHeight, int pixelSize)
        {
            int count = dataPerRow * dataPerColumn;
            double[][] input = new double[count][];
            double[][] output = new double[count][];

            Bitmap bmpInput = new Bitmap(pixelSize * dataPerRow * (dataWidth + 1) - 1, pixelSize * dataPerColumn * (dataHeight + 1) - 1);
            Bitmap bmpOutput = new Bitmap(pixelSize * dataPerRow * (dataWidth + 1) - 1, pixelSize * dataPerColumn * (dataHeight + 1) - 1);

            for (int i = 0; i < count; i++)
            {
                int j = rng.Next(0, data.ColumnCount);

                input[i] = data.Column(j).ToArray();
                output[i] = network.Update(input[i]);
            }

            input = Maths.Rescale(input, 0, 1);
            output = Maths.Rescale(output, 0, 1);

            using (Graphics gi = Graphics.FromImage(bmpInput))
            {
                using (Graphics go = Graphics.FromImage(bmpOutput))
                {
                    for (int row = 0; row < dataPerRow; row++)
                    {
                        for (int col = 0; col < dataPerColumn; col++)
                        {
                            for (int h = 0; h < dataHeight; h++)
                            {
                                for (int w = 0; w < dataWidth; w++)
                                {
                                    int valueInput = (int)(input[row + col * dataPerRow][w + h * dataWidth] * 255);
                                    int valueOutput = (int)(output[row + col * dataPerRow][w + h * dataWidth] * 255);

                                    gi.FillRectangle(new SolidBrush(Color.FromArgb(valueInput, valueInput, valueInput)),
                                        new Rectangle((w + col * (dataWidth + 1)) * pixelSize, (h + row * (dataHeight + 1)) * pixelSize, pixelSize, pixelSize));
                                    go.FillRectangle(new SolidBrush(Color.FromArgb(valueOutput, valueOutput, valueOutput)),
                                        new Rectangle((w + col * (dataWidth + 1)) * pixelSize, (h + row * (dataHeight + 1)) * pixelSize, pixelSize, pixelSize));
                                }
                            }
                        }
                    }
                }
            }

            bmpInput.Save("input.bmp");
            bmpOutput.Save("output.bmp");
        }

        public static void ExportFiltersToBitmap(Network network, int filtersPerRow, int filterWidth, int filterHeight, int pixelSize)
        {
            double[][] weights;
            double[][] normalisedMaxActivation;
            int[][] pixelNormalisedMaxActivation;
            int noFilters = network[0].NeuronCount;
            int filtersPerColumn = noFilters / filtersPerRow;
            int value, row, column;

            Bitmap bmp = new Bitmap(pixelSize * filtersPerRow * (filterWidth + 1) - 1, pixelSize * filtersPerColumn * (filterHeight + 1) - 1);

            weights = new double[noFilters][];

            for (int i = 0; i < noFilters; i++)
            {
                weights[i] = network[0][i].Weights;
            }

            normalisedMaxActivation = Maths.Rescale(ComputeMaximumActivationInput(weights), 0, 1);
            pixelNormalisedMaxActivation = normalisedMaxActivation.Select(i => i.Select(j => (int)(j * 255)).ToArray()).ToArray();

            using (Graphics g = Graphics.FromImage(bmp))
            {
                for (int i = 0; i < noFilters; i++)
                {
                    column = i % filtersPerRow;
                    row = i / filtersPerRow;

                    for (int h = 0; h < filterHeight; h++)
                    {
                        for (int w = 0; w < filterWidth; w++)
                        {
                            value = pixelNormalisedMaxActivation[i][h * filterWidth + w];

                            g.FillRectangle(new SolidBrush(Color.FromArgb(value, value, value)),
                                new Rectangle((w + column * (filterWidth + 1)) * pixelSize, (h + row * (filterHeight + 1)) * pixelSize, pixelSize, pixelSize));
                        }
                    }
                }
            }

            bmp.Save("filters.bmp");
        }
    }
}
