namespace ArtificialNeuralNetwork.Models.AnnFunctions;

public class FeedForwardFunctions
{
  public NeuralNetwork FeedForward ( NeuralNetwork nn, Vector<double> inputRow )
    {
        for (int i = 0; i < nn.LayersList[0].NeuronsList.Count; i++)
        {
            nn.LayersList[0].NeuronsList[i].Input = inputRow[i];
            nn.LayersList[0].NeuronsList[i].Output = inputRow[i];
        }

        for (int layerIndex = 1; layerIndex < nn.LayersList.Count; layerIndex++)
        {
            var previousLayer = nn.LayersList[layerIndex - 1];
            var currentLayer = nn.LayersList[layerIndex];
            for (int neuronIndex = 0; neuronIndex < currentLayer.NeuronsList.Count; neuronIndex++)
            {
                var neuron = currentLayer.NeuronsList[neuronIndex];
                neuron.Input = 0;
                for (int prevIndex = 0; prevIndex < previousLayer.NeuronsList.Count; prevIndex++)
                {
                    var prevNeuron = previousLayer.NeuronsList[prevIndex];
                    neuron.Input += prevNeuron.Output * prevNeuron.ConnectionsList[neuronIndex].Weight;
                }
                neuron.Input += neuron.Bias;
                neuron.Output = neuron.Activation.Activate(neuron.Input);
                if (currentLayer.IsOutputLayer) nn.Prediction = neuron.Output;
            }
        }
        return nn;
    }
    }