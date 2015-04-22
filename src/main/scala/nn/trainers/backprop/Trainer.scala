package nn.trainers.backprop

import nn.NeuralNetwork
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

  def train(nn: NeuralNetwork, trainingSet: DataSet) {
    val derivatives = new Gradients(nn)

    trainingSet.miniBatches(miniBatchSize).grouped(numParallel).take(numIterations).zipWithIndex.foreach {
      case (batches, iteration) =>

        evalIteration(iteration, nn, trainingSet, batches(0))

        val sumGradients = batches.par.map { data => derivatives.errorGradients(data, nn.propagate(data.features)) }.reduce {
          (sumGradients, gradients) =>
            sumGradients.zip(gradients).map {
              case (sumGradient, gradient) => sumGradient.addi(gradient)
            }
        }
        val gradients = sumGradients.map { _.muli(1D / numParallel) }

        updateWeights(nn, iteration, gradients)
    }
  }

  def updateWeights(nn: NeuralNetwork, iteration: Int, gradients: Seq[DoubleMatrix]) {
    nn.layers.zip(gradients).zip(nn.momentums).foreach {
      case ((layer, gradient), momentum) =>
        val delta = momentum.muli(momentumMultiplier).subi(gradient)
        layer.weights.addi(delta.mul(learningRate(iteration)))
    }
  }
}