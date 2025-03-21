using ArtificialNeuralNetwork.Models.ActivationFunctions;

namespace ArtificialNeuralNetwork.Models;

public class Neuron
{
    public Guid Id { get; set; }
    public double Input { get; set; }
    public double Output { get; set; }
    public double Bias { get; set; }
    public double LocalDelta { get; set; }
    public List<Connection> ConnectionsList { get; set; }
    public IActivationFunction Activation { get; set; }

    public Neuron ( IActivationFunction activationFunc, int neuronsInNextLayerCount )
    {
        Id = Guid.NewGuid();
        Activation = activationFunc;
        Bias = 0;
        ConnectionsList = new List<Connection>();
        for (int i = 0; i < neuronsInNextLayerCount; i++)
        {
            ConnectionsList.Add(new Connection());
        }
    }
    }