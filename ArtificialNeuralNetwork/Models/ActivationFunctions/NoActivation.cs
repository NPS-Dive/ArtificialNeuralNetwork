namespace ArtificialNeuralNetwork.Models.ActivationFunctions;

public class NoActivation : IActivationFunction
    {
    public double Activate ( double x )
        {
        return x;
        }

    public double Derivative ( double x )
        {
        return 1.0;
        }
    }