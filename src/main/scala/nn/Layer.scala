package nn

import nn.fn.ActivationFunction
import org.jblas.DoubleMatrix

object Layer {
  implicit def layer2LayerList(layer: Layer) = new LayerList(List(layer))

  implicit def layerList2list(list: LayerList) = list.layers

  def apply(numInputs: Int, numOutputs: Int, activation: ActivationFunction): Layer =
    new Layer(randomWeights(numInputs, numOutputs), activation)

  def apply(numOutputs: Int, activation: ActivationFunction): PartialLayer =
    new PartialLayer(numOutputs, activation)

  def randomWeights(numInputs: Int, numOutputs: Int, absMax: Double = 0.01): DoubleMatrix =
    DoubleMatrix.rand(numOutputs, numInputs).muli(2).subi(1).muli(absMax).transpose
}

class Layer(val weights: DoubleMatrix, val activation: ActivationFunction) extends Serializable {
  lazy val numInputs = weights.rows
  lazy val numOutputs = weights.columns

  def composition(x: DoubleMatrix): DoubleMatrix = weights.transpose.mmul(x)

  def apply(x: DoubleMatrix): LayerState = {
    val c = composition(x)

    LayerState(Some(c), activation(c))
  }

  def copy(weights: DoubleMatrix = this.weights, activation: ActivationFunction = this.activation): Layer =
    new Layer(weights, activation)
 }