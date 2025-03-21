namespace ArtificialNeuralNetwork.Models.AnnFunctions;

public class ConnectNeuralNetworkFunctions
{
   public NeuralNetwork ConnectNeuralNetwork ( NeuralNetwork nn )
    {
        nn.LayersList.Take(nn.LayersList.Count - 1).Select(( layer, index ) =>
        {
            layer.NeuronsList.ForEach(neuron =>
            {
                neuron.ConnectionsList.Select(( connection, connectionIndex ) =>
                {
                    connection.TargetedNeuron = nn.LayersList[index + 1].NeuronsList[connectionIndex];
                    return connection;
                }).ToList();
            });
            return layer;
        }).ToList();
        return nn;
    }
    }