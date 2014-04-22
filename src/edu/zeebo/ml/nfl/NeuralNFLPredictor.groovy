package edu.zeebo.ml.nfl
import org.encog.engine.network.activation.ActivationSigmoid
import org.encog.ml.CalculateScore
import org.encog.ml.data.MLData
import org.encog.ml.data.MLDataPair
import org.encog.ml.data.basic.BasicMLDataSet
import org.encog.ml.train.MLTrain
import org.encog.neural.networks.BasicNetwork
import org.encog.neural.networks.layers.BasicLayer
import org.encog.neural.networks.training.TrainingSetScore
import org.encog.neural.networks.training.anneal.NeuralSimulatedAnnealing
import org.encog.persist.EncogDirectoryPersistence
import org.encog.util.Format
/**
 * User: Eric
 * Date: 4/21/14
 */
class NeuralNFLPredictor {

	def allData = [:]

	def inputData = [:]
	def testData = [:]

	def inputFactors = ['stadium', 'surface', 'temperature', 'humidity', 'wind',
			'week', 'winner record', 'loser record', 'home', 'away', 'winner', 'loser', 'vegas line'
	]

	def teams = []

	def inputDataset
	def testDataset

	BasicNetwork network

	NeuralNFLPredictor(def hiddenLayerNodes) {

		def team = 'Houston Texans'

		readData(team)
		inputDataset = convertInputDataToDataset(inputData, team)
		testDataset = convertInputDataToDataset(testData, team)

		network = new BasicNetwork()
		network.addLayer(new BasicLayer(null, true, 3))
		network.addLayer(new BasicLayer(new ActivationSigmoid(), true, hiddenLayerNodes))
		network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 1))
		network.structure.finalizeStructure()
	}

	NeuralNFLPredictor(BasicNetwork network) {
		this.network = network

		def team = 'Houston Texans'

		readData(team)
		inputDataset = convertInputDataToDataset(inputData, team)
		testDataset = convertInputDataToDataset(testData, team)
	}

	def readData(def team) {
		def lines = new File('nfl_game_data.csv').readLines()

		def headers = lines[0].split(',') as List

		for (int i = 1; i < lines.size(); i++) {
			def line = lines[i].split(',') as List

			def inputs = inputFactors.collectEntries {
				[it, line[headers.indexOf(it)]]
			}

			try { inputs.week as int } catch (Exception e) {continue}

			if (teams.indexOf(inputs.home) < 0) {
				teams << inputs.home
			}
			if (teams.indexOf(inputs.away) < 0) {
				teams << inputs.away
			}
			allData[line[0]] = inputs
		}

		int weekCutoff = 14

		allData = allData.findAll { it.value.home == team || it.value.away == team }

		inputData = allData.findAll {
			(it.value.week as int) < weekCutoff
		}
		testData = allData.findAll {
			(it.value.week as int) >= weekCutoff
		}
	}

	def winFactor = 3

	def convertInputDataToDataset(def rawIn, def team) {
		def inputs = []
		def ideals = []

		rawIn.values().each {

			/*if (it.winner == team) {
				ideals << [1, 0]
			}
			else {
				ideals << [0, 1]
			}*/

			ideals << [it.winner == team ? 1 : 0]

			def winRecord = it.'winner record'.replaceAll(/[\[\]]/, '').split('\\|')
			def winnerValue = (winRecord[0] as int) * 3 - (winRecord[1] as int) + (winRecord[2] as int) * 1.5
			def loseRecord = it.'loser record'.replaceAll(/[\[\]]/, '').split('\\|')
			def loserValue = (loseRecord[0] as int) * 3 - (loseRecord[1] as int) + (loseRecord[2] as int) * 1.5

			def temp = it.temperature
			def humidity = it.humidity.replace('%', '')
			def wind = it.wind
			def home = teams.indexOf(it.home)
			def away = teams.indexOf(it.away)

			temp = (temp ?: '0') as double
			humidity = (humidity ?: '0') as double
			wind = (wind ?: '0') as double

			def vl = it.'vegas line'
			if (vl.empty || vl == 'Pick') {
				vl = 0
			}
			else {
				def vlTeam = vl[(0..<vl.lastIndexOf(' '))]
				vl = (vl.replace(vlTeam, '').trim() as double) * (vlTeam == team ? 1 : -1)
			}

//			inputs << [temp, humidity, wind, winnerValue, loserValue, vl]
			inputs << [winnerValue, loserValue, vl]
		}

		return new BasicMLDataSet(inputs as double[][], ideals as double[][])
	}

	def evaluate() {
		def vals = []
		testDataset.each { MLDataPair pair ->
			MLData output = network.compute(pair.input)
			println "" + pair.input + ", Actual=" + output + ", Ideal=" + pair.ideal
			vals << [pair.ideal.data[0], output.data[0]]
		}

		return calculateError(vals)
	}

	def calculateError(List vals) {
		def epsilon = 1e-15

		double logLoss = vals.sum {
			def pred = Math.max(epsilon, Math.min(it[1], 1 - epsilon))
			def act  = it[0]
			return act * Math.log(pred) + (1 - act) * Math.log(1 - pred)
		}
		return -logLoss / vals.size()
	}

	def train(error, iterations, startTemp, endTemp) {

		network.reset()

		CalculateScore score = new TrainingSetScore(testDataset)
		MLTrain train = new NeuralSimulatedAnnealing(network, score, startTemp, endTemp, 500)

		def epoch = 0
		train.iteration()

		while (epoch < iterations && train.error > error && !train.trainingDone) {
			train.iteration()
			epoch++
		}
		train.finishTraining()

		println "Iteration #" + Format.formatInteger(epoch) +
				" Error:" + Format.formatPercent(train.error) +
				" Target Error: " + Format.formatPercent(error)

		return [epoch >= iterations, train.error]
	}

	def save(def name) {
		EncogDirectoryPersistence.saveObject(new File("${name}.nn"), network)
	}

	public static void main(String[] args) {
		findBestNodeCount(1, 35)
//		BasicNetwork network = (BasicNetwork)EncogDirectoryPersistence.loadObject(new File('Texans/0.058899524707952795.nn'))
//		NeuralNFLPredictor nfl = new NeuralNFLPredictor(network)
//		println nfl.evaluate()
	}

	static def nodeCountToError = [:]
	static int currentThreads = 0

	static def findBestNodeCount(int min, int max) {

		int nextNumber = min

		int maxThreads = 5

		int startTemp = 50.57686066071585
		int endTemp = 58.51986168332346

		while (nextNumber <= max) {
//		for (int i = 0; i < 10; i++) {
			if (currentThreads < maxThreads) {
				currentThreads++
				Thread.start {
					def num = nextNumber++
//					def num = i
					NeuralNFLPredictor nfl = new NeuralNFLPredictor(num)
					def st = startTemp// * Math.random()
					def et = endTemp// * Math.random()
					def res = nfl.train(0.01, 100, st, et)

					def err = nfl.evaluate()

					nodeCountToError[num] = [err, st, et, nfl]
					finishThread(num)
				}
			}
//			else {
//				i--
//			}
			sleep 10
		}

		while (currentThreads > 0) {
			sleep 10
		}

		def list = nodeCountToError.keySet().collect { [nodeCountToError[it], it] }
		list.sort { a, b -> a[0][0] <=> b[0][0] }

		println list

		list[0][0][3].save(list[0][0][0])
	}

	static def finishThread(def num) {
		println "Finished $num"
		currentThreads--
	}
}
