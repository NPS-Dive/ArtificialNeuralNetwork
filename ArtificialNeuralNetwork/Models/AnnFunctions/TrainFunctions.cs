namespace ArtificialNeuralNetwork.Models.AnnFunctions;

public class TrainFunctions
{
    public NeuralNetwork Train ( NeuralNetwork nn, int epochs, double learningRate, Matrix<double> inputs, Vector<double> outputs )
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int row = 0; row < inputs.RowCount; row++)
            {
                nn = new FeedForwardFunctions().FeedForward(nn, inputs.Row(row));
                nn = new BackwardPassFunctions(). BackwardPass(nn, outputs[row]);
                nn = new WeightAdjustmentFunctions().WeightAdjustment(nn, learningRate);
            }
        }
        return nn;
    }
    }