namespace ArtificialNeuralNetwork.Models.AnnFunctions;

public class WeightAdjustmentFunctions
{
    public NeuralNetwork WeightAdjustment ( NeuralNetwork nn, double learningRate )
    {
        nn.LayersList.ForEach(layer =>
        {
            layer.NeuronsList.ForEach(neuron =>
            {
                if (!layer.IsOutputLayer)
                {
                    if (!layer.IsInputLayer) neuron.Bias -= learningRate * neuron.LocalDelta;
                    neuron.ConnectionsList.ForEach(c =>
                        c.Weight -= learningRate * c.TargetedNeuron.LocalDelta * neuron.Output);
                }
            });
        });
        return nn;
    }
    }