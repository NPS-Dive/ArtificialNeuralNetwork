namespace ArtificialNeuralNetwork.Requests;

public class TrainRequest
{
    public double[,] Inputs { get; set; }
    public double[] Outputs { get; set; }
    public int Epochs { get; set; }
    public double LearningRate { get; set; }
}