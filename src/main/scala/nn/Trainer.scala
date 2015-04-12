package nn

import nn.ds.DataSet
import nn.fn.LearningFunction
import nn.fn.lrn.ConstantRate
import org.jblas.DoubleMatrix

object Trainer {

  def apply(numIterations: Int, miniBatchSize: Int = 100, learningRate: LearningFunction = ConstantRate(0.5),
            momentumMultiplier: Double = 1.0, evalIterations: Int = 100, numParallel: Int = 1) =
    new Trainer(numIterations, miniBatchSize, if (numParallel < 1) 1 else numParallel, learningRate, momentumMultiplier, evalIterations)

}

class Trainer(val numIterations: Int, val miniBatchSize: Int, val numParallel: Int, val learningRate: LearningFunction,
              val momentumMultiplier: Double, val evalIterations: Int) {

  def evalIteration(iteration: Int, network: NeuralNetwork, trainingSet: DataSet, batch: DataSet) {
    if ((iteration + 1) % evalIterations == 0) {
      val loss = network.loss(trainingSet)
      println("Iteration:%5d, Loss: %.5f, Accuracy: %f".format(iteration + 1, loss, network.eval(trainingSet)))
    }
  }

  def train(network: NeuralNetwork, trainingSet: DataSet) {

    val momentums = network.weights.map { _.mul(0) }

    trainingSet.miniBatches(miniBatchSize).grouped(numParallel).take(numIterations).zipWithIndex.foreach {
      case (batches, iteration) =>

        evalIteration(iteration, network, trainingSet, batches(0))

        val batchGradients = batches.par.map { network.errorGradients(_) }
        val sumGradients = batchGradients.reduce {
          (sumGradients, gradients) =>
            sumGradients.zip(gradients).map {
              case (sumGradient, gradient) => sumGradient.addi(gradient)
            }
        }
        val gradients = sumGradients.map { _.muli(1D / numParallel) }

        updateWeights(network, iteration, gradients, momentums)
    }
  }

  def updateWeights(network: NeuralNetwork, iteration: Int, gradients: Seq[DoubleMatrix], momentums: Seq[DoubleMatrix]) {
    network.layers.zip(gradients).zip(momentums).foreach {
      case ((layer, gradient), momentum) =>
        val delta = momentum.muli(momentumMultiplier).subi(gradient)
        layer.weights.addi(delta.mul(learningRate(iteration)))
    }
  }
}