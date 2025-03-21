namespace ArtificialNeuralNetwork.Models.ActivationFunctions;

public class TanH : IActivationFunction
    {
    public double Activate ( double x )
        {
        return Math.Tanh(x);
        }

    public double Derivative ( double x )
        {
        var tanh = Activate(x);
        return 1.0 - (tanh * tanh);
        }
    }