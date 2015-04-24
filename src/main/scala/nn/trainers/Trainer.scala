package nn.trainers

import nn.NeuralNetwork
import nn.ds.DataSet

trait Trainer[T <: NeuralNetwork] {
  def train(trainingSet: DataSet): T

  def evalIterations: Int

  def nn:T

  def evalIteration(iteration: Int, trainingSet: DataSet) {
    if ((iteration + 1) % evalIterations == 0) {
      val loss = nn.loss(trainingSet)

      println("Iteration:%5d, Loss: %.5f, Accuracy: %f".format(iteration + 1, loss, nn.eval(trainingSet)))
    }
  }
}
