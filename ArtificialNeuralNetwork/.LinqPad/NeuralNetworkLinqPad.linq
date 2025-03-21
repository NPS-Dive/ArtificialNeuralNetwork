<Query Kind="Program">
  <NuGetReference>MathNet.Numerics</NuGetReference>
  <Namespace>MathNet.Numerics</Namespace>
  <Namespace>MathNet.Numerics.LinearAlgebra</Namespace>
  <IncludePredicateBuilder>true</IncludePredicateBuilder>
  <IncludeLinqToSql>true</IncludeLinqToSql>
  <DisableMyExtensions>true</DisableMyExtensions>
  <CopyLocal>true</CopyLocal>
  <CreateRuntimesFolder>true</CreateRuntimesFolder>
  <RuntimeVersion>8.0</RuntimeVersion>
</Query>

//nuget: MathNet.Numerics

using System;

void Main()
{
	var myNeuralNetwork = CreateNeuralNetwork(new List<int>() { 2, 8, 8, 1 });
	myNeuralNetwork = ConnectNeuralNetwork(myNeuralNetwork);

	var inputs = Matrix<double>.Build.DenseOfArray(new double[,] {
	{1,2},
	{-2,5},
	{3,7},
	{4,8},
	{-4,7},
	{6,14},
	{7,13},
	{8,7},
	{12,10},
	{17,16},
	{18,22},
	});

	var outputs = Vector<double>.Build.DenseOfEnumerable(new List<double>() {
	1*1+2*2,
	2*2+5*5,
	3*3,7*7,
	4*4+8*8,
	4*4+7*7,
	6*6+14*14,
	7*7+13*13,
	8*8,7*7,
	12*12+10*10,
	17*17+16*16,
	18*18+22*22,
	});

	#region
	
	//nornalization of inputs
	var inputMax = inputs.Enumerate().Max();
	inputs = inputs.Map(i => i / inputMax);

	//normalization of outputs
	var outputMax = outputs.Enumerate().Max();
	outputs = outputs.Map(o => o / outputMax);
	
	#endregion

	var epochs = 60000;

	double learningRate = 0.1;

	var graphData = new GraphData();
	graphData.Data = new List<Data>();

	myNeuralNetwork = Train(myNeuralNetwork, epochs, learningRate, inputs, outputs);

	//NAND function
	myNeuralNetwork = Predict(myNeuralNetwork, Vector<double>.Build.DenseOfArray(new double[] { 5/inputMax, 9/inputMax }));

//denormalize the normalized predication	
var prediction= myNeuralNetwork.Prediction*outputMax;


(5 * 6 + 9 * 9).Dump("the correct answer");
prediction.Dump("Prediction");

	
	// internal graph of LinqPad
	//graphData.Data.Chart(c => c.Epoch, c => c.Loss, Util.SeriesType.Line).Dump();


	NeuralNetwork Predict(
	NeuralNetwork neuralNetwork,
	Vector<double> input)
	{
		neuralNetwork = FeedForward(neuralNetwork, input);
		return neuralNetwork;
	}

	NeuralNetwork Train(
	NeuralNetwork neuralNetwork,
	int epochCount,
	double learningRate,
	Matrix<double> inputs,
	Vector<double> output)
	{
		for (int epoch = 0; epoch < epochCount; epoch++)
		{
			for (int trainRowIndex = 0; trainRowIndex < inputs.RowCount; trainRowIndex++)
			{
				neuralNetwork = FeedForward(neuralNetwork, inputs.Row(trainRowIndex));
				neuralNetwork = BackwardPass(neuralNetwork, outputs[trainRowIndex]);
				neuralNetwork = WeightAdjustment(neuralNetwork, learningRate);
			}
			graphData.Data.Add(new Data
			{
				Loss = neuralNetwork.CurrentLoss,
				Epoch = epoch
			});
		}
		return neuralNetwork;
	}


	NeuralNetwork CreateNeuralNetwork(List<int> neuronsInEachLayerCount)
	{
		return new NeuralNetwork(neuronsInEachLayerCount);
	}

	NeuralNetwork ConnectNeuralNetwork(NeuralNetwork nn)
	{
		nn.LayersList.Take(nn.LayersList.Count - 1).Select((layer, index) =>
		{
			layer.NeuronsList.ForEach(neuron =>
			{
				neuron.ConnectionsList.Select((connection, connectionIndex) =>
				{
					connection.TargetedNeuron = nn
					.LayersList[index + 1]
					.NeuronsList[connectionIndex];

					return connection;
				}).ToList();
			});
			return layer;
		}).ToList();

		return nn;
	}

	NeuralNetwork FeedForward(NeuralNetwork neuralnetwork, Vector<double> inputRow)
	{
		// see intput values for the first layer
		for (int i = 0; i < neuralnetwork.LayersList[0].NeuronsList.Count; i++)
		{
			neuralnetwork.LayersList[0].NeuronsList[i].Input = inputRow[i];
			neuralnetwork.LayersList[0].NeuronsList[i].Output = inputRow[i];
		}

		for (int layerIndex = 1; layerIndex < neuralnetwork.LayersList.Count; layerIndex++)
		{
			var previousLayer = neuralnetwork.LayersList[layerIndex - 1];
			var currentLayer = neuralnetwork.LayersList[layerIndex];

			for (int neuronIndex = 0; neuronIndex < currentLayer.NeuronsList.Count; neuronIndex++)
			{
				var neuron = currentLayer.NeuronsList[neuronIndex];
				neuron.Input = 0;

				//claculate the weightedSum of the3 given neuron upon the output of the previous layers 
				for (int previousLayerNeuronIndex = 0; previousLayerNeuronIndex < previousLayer.NeuronsList.Count; previousLayerNeuronIndex++)
				{
					var previousNeuron = previousLayer.NeuronsList[previousLayerNeuronIndex];
					neuron.Input += previousNeuron.Output * previousNeuron.ConnectionsList[neuronIndex].Weight;
				}
				neuron.Input += neuron.Bias;
				neuron.Output = neuron.Activation.Activate(neuron.Input);

				if (currentLayer.IsOutputLayer)
				{
					neuralnetwork.Prediction = neuron.Output;
				}
			}
		}

		return neuralnetwork;
	}

	NeuralNetwork BackwardPass(NeuralNetwork neuralNetwork, double correctOutput)
	{
		//derivative of output neuron 
		var outputNeuron = neuralNetwork.LayersList.Last().NeuronsList.First();

		neuralNetwork.CurrentLoss = Math.Pow(correctOutput - neuralNetwork.Prediction, 2);

		//mean-square error (MSE) derivative
		var lossDeravative = 2 * (neuralNetwork.Prediction - correctOutput);

		var outputActivationFunctionDerivative = outputNeuron.Activation.Derivative(outputNeuron.Input);
		var localDelta = lossDeravative * outputActivationFunctionDerivative;
		outputNeuron.LocalDelta = localDelta;

		//rest of the neuron
		neuralNetwork.LayersList.Reverse();
		neuralNetwork.LayersList.Skip(1).ToList().ForEach(layer =>
		{
			layer.NeuronsList.ForEach(neuron =>
			{
				//claculate the local delta(error gradiant) upon deltas of next-layer neurons
				neuron.LocalDelta = neuron
				.ConnectionsList
				.Sum(connection => connection.Weight * connection.TargetedNeuron.LocalDelta);

				neuron.LocalDelta = neuron.LocalDelta * neuron.Activation.Derivative(neuron.Input);
			});
		});
		neuralNetwork.LayersList.Reverse();
		return neuralNetwork;
	}

	NeuralNetwork WeightAdjustment(NeuralNetwork neuralNetwork, double learningRate)
	{
		neuralNetwork.LayersList.ToList().ForEach(layer =>
		{
			layer.NeuronsList.ForEach(neuron =>
			{
				if (layer.IsOutputLayer is false)
				{
					if (layer.IsInputLayer is false)
					{
						neuron.Bias -= learningRate * neuron.LocalDelta;
					}
					neuron.ConnectionsList.ForEach(connection =>
					{
						connection.Weight -= (learningRate * connection.TargetedNeuron.LocalDelta * neuron.Output);
					});
				}
			});
		});
		return neuralNetwork;
	}
}

public class NeuralNetwork
{
	public List<Layer> LayersList { get; set; }
	public double Prediction { get; set; }
	public ILossFunctions Loss { get; set; }
	public double CurrentLoss { get; set; }

	#region Counstructor
	public NeuralNetwork(List<int> neuronsInEachLayerCount)
	{
		LayersList = new List<Layer>();

		LayersList = neuronsInEachLayerCount
		.Select((neuronsInLayerCount, index) =>
		{
			//determining neurons in the next layer
			var neuronsInNextLayerCount = (index < (neuronsInEachLayerCount.Count - 1))
			? neuronsInEachLayerCount[index + 1]
			: 0;

			bool isInputLayer;
			if (index == 0)
			{
				isInputLayer = true;
			}
			else
			{
				isInputLayer = false;
			}

			return new Layer(neuronsInLayerCount, neuronsInNextLayerCount, isInputLayer);
		})
		.ToList();

	}
	#endregion
}

#region Classes

public class Layer
{
	public List<Neuron> NeuronsList { get; set; }
	public bool IsInputLayer { get; set; }
	public bool IsOutputLayer { get; set; }

	#region Counstructor
	public Layer(int neuronsInLayerCount, int neuronsInNextLayerCount, bool isInputLayer)
	{
		if (neuronsInNextLayerCount == 0)
		{
			IsOutputLayer = true;
		}

		IsInputLayer = isInputLayer;

		IActivationFunction activation = IsOutputLayer
		? new NoActivation()
		: new TanH();

		activation = IsInputLayer
		? new NoActivation()
		: activation;

		NeuronsList = new List<Neuron>();

		for (int i = 0; i < neuronsInLayerCount; i++)
		{
			NeuronsList.Add(new Neuron(activation, neuronsInNextLayerCount));
		}


	}
	#endregion
}

public class Neuron
{
	public Guid Id { get; set; }
	public double Input { get; set; }
	public double Output { get; set; }
	public double Bias { get; set; }
	public double LocalDelta { get; set; }
	public List<Connection> ConnectionsList { get; set; }
	public IActivationFunction Activation { get; set; }

	#region Constructor
	public Neuron(IActivationFunction activationFunc, int neuronsInNextLayerCount)
	{
		Id = Guid.NewGuid();
		Activation = activationFunc;
		Bias = 0;
		ConnectionsList = new List<Connection>();

		for (int i = 0; i < neuronsInNextLayerCount; i++)
		{
			ConnectionsList.Add(new Connection());
		}
	}
	#endregion
}

public class Connection
{
	public double Weight { get; set; }
	public Neuron TargetedNeuron { get; set; }

	#region Counstructor
	public Connection()
	{
		var random = new Random();
		Weight = random.NextDouble();
	}
	#endregion
}

public class LogisticSigmuid : IActivationFunction
{
	public double Activate(double x)
	{
		return 1.0 / (1.0 + Math.Exp(-x));
	}

	public double Derivative(double x)
	{
		var sigmuid = Activate(x);
		var resultDerivative = sigmuid * (1.0 - sigmuid);
		return resultDerivative;
	}
}

public class TanH : IActivationFunction
{
	public double Activate(double x)
	{
		//compressing the input which received into Output between -1 & 1 (implemented in hidden layers)
		return Math.Tanh(x);
	}

	public double Derivative(double x)
	{
		var tanhH = Activate(x);
		var resultDerivative = 1.0 - (tanhH * tanhH);
		return resultDerivative;
	}
}

public class NoActivation : IActivationFunction
{
	public double Activate(double x)
	{
		return x;
	}
	public double Derivative(double x)
	{
		return 1.0;
	}
}

public class GraphData
{
	public List<Data> Data { get; set; }
}

public class Data
{
	public double Epoch { get; set; }
	public double Loss { get; set; }
}

#endregion



#region Interfaces
public interface ILossFunctions
{

}

public interface IActivationFunction
{
	public double Activate(double x);
	double Derivative(double input);
}
#endregion
