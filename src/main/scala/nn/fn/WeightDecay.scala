package nn.fn

import nn.Layer
import org.jblas.DoubleMatrix
import org.jblas.MatrixFunctions._

object WeightDecay {
  def apply(coefficient: Double): WeightDecay = new WeightDecay(coefficient)

  val ZERO:WeightDecay = WeightDecay(0.0)
}

class WeightDecay(coefficient: Double) {
  def apply(layers: List[Layer], loss: Double): Double =
    loss + (layers.map{ l => pow(l.weights,2).sum }.sum / 2 * coefficient)

  def derivative(layer: Layer, layerDerivative: DoubleMatrix): DoubleMatrix =
    layerDerivative.add(layer.weights.mul(coefficient))
}