namespace ArtificialNeuralNetwork.Models.ActivationFunctions;

public interface IActivationFunction
{
    double Activate ( double x );
    double Derivative ( double input );
    }