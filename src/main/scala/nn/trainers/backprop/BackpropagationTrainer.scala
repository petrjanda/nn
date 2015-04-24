package nn.trainers.backprop

import nn.trainers.Trainer
import nn.{FeedForwardNN}
import nn.ds.DataSet
import nn.fn.LearningFunction
import nn.fn.lrn.ConstantRate
import org.jblas.DoubleMatrix

object BackpropagationTrainer {

  def apply(nn:FeedForwardNN, numIterations: Int, miniBatchSize: Int = 100, learningRate: LearningFunction = ConstantRate(0.5),
            momentumMultiplier: Double = 1.0, evalIterations: Int = 100, numParallel: Int = 1) =
    new BackpropagationTrainer(nn, numIterations, miniBatchSize, if (numParallel < 1) 1 else numParallel, learningRate, momentumMultiplier, evalIterations)

}

class BackpropagationTrainer(val nn:FeedForwardNN, val numIterations: Int, val miniBatchSize: Int, val numParallel: Int, val learningRate: LearningFunction,
              val momentumMultiplier: Double, val evalIterations: Int) extends Trainer[FeedForwardNN] {

  def train(trainingSet: DataSet) = {
    val derivatives = new Gradients(nn)

    trainingSet.miniBatches(miniBatchSize).grouped(numParallel).take(numIterations).zipWithIndex.foreach {
      case (batches, iteration) =>

        evalIteration(iteration, trainingSet)

        val sumGradients = batches.par.map { data => derivatives.errorGradients(data, nn.propagate(data.features)) }.reduce {
          (sumGradients, gradients) =>
            sumGradients.zip(gradients).map {
              case (sumGradient, gradient) => sumGradient.addi(gradient)
            }
        }
        val gradients = sumGradients.map { _.muli(1D / numParallel) }

        nn.updateWeights(learningRate(iteration), gradients)
    }

    nn
  }
}