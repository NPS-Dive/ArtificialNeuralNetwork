namespace ArtificialNeuralNetwork.Requests;

public class PredictRequest
{
    public double[] Input { get; set; }
    public double InputMax { get; set; }
    public double OutputMax { get; set; }
}