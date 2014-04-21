package edu.zeebo.ml.nfl

import org.encog.engine.network.activation.ActivationSigmoid
import org.encog.ml.CalculateScore
import org.encog.ml.data.MLDataSet
import org.encog.ml.data.basic.BasicMLDataSet
import org.encog.neural.data.NeuralDataSet
import org.encog.neural.networks.BasicNetwork
import org.encog.neural.networks.layers.BasicLayer
import org.encog.neural.networks.training.TrainingSetScore
import org.encog.neural.networks.training.anneal.NeuralSimulatedAnnealing

/**
 * User: Eric
 * Date: 4/21/14
 */
class NeuralNFLPredictor {

	def inputData = [:]

	def testData = [:]

	def inputFactors = ['stadium', 'surface', 'temperature', 'humidity', 'wind',
			'week', 'winner record', 'loser record', 'home', 'away', 'winner', 'loser']

	NeuralNFLPredictor() {
		BasicNetwork network = new BasicNetwork()
		network.addLayer(new BasicLayer(null, true, 2))
		network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 3))
		network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 1))
		network.structure.finalizeStructure()
		network.reset()

		MLDataSet data = new BasicMLDataSet()
		CalculateScore score = new TrainingSetScore()

		NeuralSimulatedAnnealing training = new NeuralSimulatedAnnealing(network, )
	}

	def readData() {
		def lines = new File('nfl_game_data.csv').readLines()

		def headers = lines[0].split(',')

		for (int i = 1; i < lines.size(); i++) {
			def line = lines[i].split(',')
		}
	}

	public static void main(String[] args) {
		new NeuralNFLPredictor()
	}
}
