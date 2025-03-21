using ArtificialNeuralNetwork.Models.AnnFunctions;
using ArtificialNeuralNetwork.Requests;

namespace ArtificialNeuralNetwork.Controllers
    {
    [ApiController]
    [Route("api/[controller]/[action]")]
    public class ArtificialNeuralNetworkController : ControllerBase
        {
        [HttpPost()]
        public IActionResult Train ( [FromBody] TrainRequest request )
            {
            var nn = new NeuralNetwork(new List<int> { 2, 8, 8, 1 });
            nn = new ConnectNeuralNetworkFunctions().ConnectNeuralNetwork(nn);

            var inputs = Matrix<double>.Build.DenseOfArray(request.Inputs);
            var outputs = Vector<double>.Build.DenseOfArray(request.Outputs);

            // Normalization
            var inputMax = inputs.Enumerate().Max();
            var inputsNormalized = inputs.Map(i => i / inputMax);
            var outputMax = outputs.Enumerate().Max();
            var outputsNormalized = outputs.Map(o => o / outputMax);

            nn = new TrainFunctions().Train(nn, request.Epochs, request.LearningRate, inputsNormalized, outputsNormalized);

            return Ok(new { Message = "Training completed", Loss = nn.CurrentLoss });
            }

        [HttpPost()]
        public IActionResult Predict ( [FromBody] PredictRequest request )
            {
            var nn = new NeuralNetwork(new List<int> { 2, 8, 8, 1 });
            nn = new ConnectNeuralNetworkFunctions().ConnectNeuralNetwork(nn);

            // Assuming the network is pre-trained; in a real app, you'd load weights
            var input = Vector<double>.Build.DenseOfArray(request.Input);
            var inputMax = request.InputMax; // Pass normalization factor
            var outputMax = request.OutputMax;

            nn = new FeedForwardFunctions().FeedForward(nn, input.Map(i => i / inputMax));
            var prediction = nn.Prediction * outputMax;

            return Ok(new { Prediction = prediction });
            }

       

       

        

      

       
        }
    }
