using ANN.Core;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;

namespace ANN.Utils
{
    public static class Networks
    {
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

        public static void ExportHiddenWeightsToBitmap(Network network, int width, int height, int wdiv, int hdiv)
        {
            if (network.LayerCount < 1)
            {
                throw new ArgumentException("The network does not contain any layers");
            }

            double[] normalisedMaxActivation;
            int[] intNormalisedMaxActivation;
            int value;
            int wstep = width / wdiv;
            int hstep = height / hdiv;
            string filename;
            Bitmap bmp;

            for (int i = 0; i < network.LayerCount - 1; i++)
            {
                for (int j = 0; j < network[i].NeuronCount; j++)
                {
                    bmp = new Bitmap(width, height);

                    normalisedMaxActivation = Maths.Rescale(ComputeMaximumActivationInput(network[i][j].Weights), 0, 1);
                    intNormalisedMaxActivation = normalisedMaxActivation.Select(n => (int)(n * 255)).ToArray();

                    using (Graphics g = Graphics.FromImage(bmp))
                    {
                        for (int h = 0; h < hdiv; h++)
                        {
                            for (int w = 0; w < wdiv; w++)
                            {
                                value = intNormalisedMaxActivation[w + h * wdiv];
                                g.FillRectangle(new SolidBrush(Color.FromArgb(value, value, value)),
                                    new Rectangle((wdiv - w - 1) * wstep, (hdiv - h - 1) * hstep, wstep, hstep));
                            }
                        }
                    }

                    filename = string.Format("layer{0}_unit{1}.bmp", i, j);
                    bmp.Save(filename);
                }
            }
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
