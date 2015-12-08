using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using LumenWorks.Framework.IO.Csv;
using System.IO;

namespace AutoencoderDemo
{
    class Program
    {
        static void Main(string[] args)
        {
            List<int[]> trainingData = new List<int[]>();
            List<int> labels = new List<int>();

            using (CsvReader csv =
                   new CsvReader(new StringReader(Properties.Resources.kaggle_train), true))
            {
                int fieldCount = csv.FieldCount;
                Console.WriteLine("Field count: {0}", fieldCount);

                string[] headers = csv.GetFieldHeaders();

                while(csv.ReadNextRecord())
                {
                    List<int> trainingExample = new List<int>();

                    labels.Add(int.Parse(csv[0]));
                    for(int i = 1; i < fieldCount; i++)
                    {
                        trainingExample.Add(int.Parse(csv[i]));
                    }

                    trainingData.Add(trainingExample.ToArray());
                }
            }

            Console.WriteLine("Total examples: {0}", trainingData.Count);
            Console.WriteLine("Total labels:   {0}", labels.Count);

        }
    }
}
