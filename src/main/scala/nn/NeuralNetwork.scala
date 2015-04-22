package nn

import nn.ds.DataSet

trait NeuralNetwork {
  def loss(data: DataSet): Double

  def eval(data: DataSet): Double
}
