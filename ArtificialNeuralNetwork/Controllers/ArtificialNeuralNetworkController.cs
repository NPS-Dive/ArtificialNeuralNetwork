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
            nn = ConnectNeuralNetwork(nn);

            var inputs = Matrix<double>.Build.DenseOfArray(request.Inputs);
            var outputs = Vector<double>.Build.DenseOfArray(request.Outputs);

            // Normalization
            var inputMax = inputs.Enumerate().Max();
            var inputsNormalized = inputs.Map(i => i / inputMax);
            var outputMax = outputs.Enumerate().Max();
            var outputsNormalized = outputs.Map(o => o / outputMax);

            nn = Train(nn, request.Epochs, request.LearningRate, inputsNormalized, outputsNormalized);

            return Ok(new { Message = "Training completed", Loss = nn.CurrentLoss });
            }

        [HttpPost("predict")]
        public IActionResult Predict ( [FromBody] PredictRequest request )
            {
            var nn = new NeuralNetwork(new List<int> { 2, 8, 8, 1 });
            nn = ConnectNeuralNetwork(nn);

            // Assuming the network is pre-trained; in a real app, you'd load weights
            var input = Vector<double>.Build.DenseOfArray(request.Input);
            var inputMax = request.InputMax; // Pass normalization factor
            var outputMax = request.OutputMax;

            nn = FeedForward(nn, input.Map(i => i / inputMax));
            var prediction = nn.Prediction * outputMax;

            return Ok(new { Prediction = prediction });
            }

        private NeuralNetwork ConnectNeuralNetwork ( NeuralNetwork nn )
            {
            nn.LayersList.Take(nn.LayersList.Count - 1).Select(( layer, index ) =>
            {
                layer.NeuronsList.ForEach(neuron =>
                {
                    neuron.ConnectionsList.Select(( connection, connectionIndex ) =>
                    {
                        connection.TargetedNeuron = nn.LayersList[index + 1].NeuronsList[connectionIndex];
                        return connection;
                    }).ToList();
                });
                return layer;
            }).ToList();
            return nn;
            }

        private NeuralNetwork FeedForward ( NeuralNetwork nn, Vector<double> inputRow )
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

        private NeuralNetwork Train ( NeuralNetwork nn, int epochs, double learningRate, Matrix<double> inputs, Vector<double> outputs )
            {
            for (int epoch = 0; epoch < epochs; epoch++)
                {
                for (int row = 0; row < inputs.RowCount; row++)
                    {
                    nn = FeedForward(nn, inputs.Row(row));
                    nn = BackwardPass(nn, outputs[row]);
                    nn = WeightAdjustment(nn, learningRate);
                    }
                }
            return nn;
            }

        private NeuralNetwork BackwardPass ( NeuralNetwork nn, double correctOutput )
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

        private NeuralNetwork WeightAdjustment ( NeuralNetwork nn, double learningRate )
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

    public class TrainRequest
        {
        public double[,] Inputs { get; set; }
        public double[] Outputs { get; set; }
        public int Epochs { get; set; }
        public double LearningRate { get; set; }
        }

    public class PredictRequest
        {
        public double[] Input { get; set; }
        public double InputMax { get; set; }
        public double OutputMax { get; set; }
        }
    }
