namespace ArtificialNeuralNetwork.Models.AnnFunctions;

public class BackwardPassFunctions
{
    public NeuralNetwork BackwardPass ( NeuralNetwork nn, double correctOutput )
    {
        var outputNeuron = nn.LayersList.Last().NeuronsList.First();
        nn.CurrentLoss = Math.Pow(correctOutput - nn.Prediction, 2);
        var lossDerivative = 2 * (nn.Prediction - correctOutput);
        var outputActivationDerivative = outputNeuron.Activation.Derivative(outputNeuron.Input);
        outputNeuron.LocalDelta = lossDerivative * outputActivationDerivative;

        nn.LayersList.Reverse();
        nn.LayersList.Skip(1).ToList().ForEach(layer =>
        {
            layer.NeuronsList.ForEach(neuron =>
            {
                neuron.LocalDelta = neuron.ConnectionsList.Sum(c => c.Weight * c.TargetedNeuron.LocalDelta);
                neuron.LocalDelta *= neuron.Activation.Derivative(neuron.Input);
            });
        });
        nn.LayersList.Reverse();
        return nn;
    }
    }