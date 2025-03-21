namespace ArtificialNeuralNetwork.Models;

public class Connection
{
    public double Weight { get; set; }
    public Neuron TargetedNeuron { get; set; }

    public Connection ()
    {
        var random = new Random();
        Weight = random.NextDouble();
    }
    }