

namespace ArtificialNeuralNetwork.Models;

public class Layer
    {
    public List<Neuron> NeuronsList { get; set; }
    public bool IsInputLayer { get; set; }
    public bool IsOutputLayer { get; set; }

    public Layer ( int neuronsInLayerCount, int neuronsInNextLayerCount, bool isInputLayer )
        {
        IsOutputLayer = neuronsInNextLayerCount == 0;
        IsInputLayer = isInputLayer;

        IActivationFunction activation = IsOutputLayer ? new NoActivation() : new TanH();
        activation = IsInputLayer ? new NoActivation() : activation;

        NeuronsList = new List<Neuron>();
        for (int i = 0; i < neuronsInLayerCount; i++)
            {
            NeuronsList.Add(new Neuron(activation, neuronsInNextLayerCount));
            }
        }
    }