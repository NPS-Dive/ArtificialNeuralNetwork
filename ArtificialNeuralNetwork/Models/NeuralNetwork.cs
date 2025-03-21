namespace ArtificialNeuralNetwork.Models;

public class NeuralNetwork
    {
    public List<Layer> LayersList { get; set; }
    public double Prediction { get; set; }
    public double CurrentLoss { get; set; }

    public NeuralNetwork ( List<int> neuronsInEachLayerCount )
        {
        LayersList = neuronsInEachLayerCount.Select(( count, index ) =>
        {
            var nextLayerCount = index < neuronsInEachLayerCount.Count - 1 ? neuronsInEachLayerCount[index + 1] : 0;
            return new Layer(count, nextLayerCount, index == 0);
        }).ToList();
        }
    }