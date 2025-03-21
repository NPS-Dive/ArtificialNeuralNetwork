
namespace ArtificialNeuralNetwork.Models.ActivationFunctions;

public class LogisticSigmoid : IActivationFunction
    {
    public double Activate ( double x )
        {
        return 1.0 / (1.0 + Math.Exp(-x));
        }

    public double Derivative ( double x )
        {
        var sigmoid = Activate(x);
        return sigmoid * (1.0 - sigmoid);
        }
    }