package nn

import nn.ds.DataSet
import nn.fn.{WeightDecay, ObjectiveFunction}
import org.jblas.DoubleMatrix

/**
 * Backpropagation algorithm for feed forward NN with optional weight decay as a regularization.
 *
 * @param layers  List of layers.
 * @param objective Objective function.
 * @param weightDecay Weight decay settings.
 */
case class Derivatives(layers:List[Layer], objective:ObjectiveFunction, weightDecay: WeightDecay) {
  /**
   * Calculate error gradient for a given dataset.
   *
   * @param data
   * @param outputs
   * @return
   */
  def errorGradients(data: DataSet, outputs:List[LayerState]): Seq[DoubleMatrix] = {
    val outErrorDerivative = objective.derivative(outputs.last.activationOutput, data.targets)

    derivatives(outErrorDerivative, outputs).zipWithIndex.map {
      case (derivative, i) =>
        val x = if (i > 0) outputs(i - 1).activationOutput else data.inputs

        decayed(i, gradient(x, derivative))
    }
  }

  private def gradient(x: DoubleMatrix, derivative: DoubleMatrix) =
    x.mmul(derivative.transpose)

  private def decayed(i:Int, gradient: DoubleMatrix) =
    if (i < layers.size) weightDecay.derivative(layers(i), gradient) else gradient

  private def derivatives(initial:DoubleMatrix, outputs:List[LayerState]) =
    (0 until layers.size).scanRight(initial) {
      case (i, priorDerivative) =>
        val priorDerivativeWeighted = if (i < layers.size - 1) {
          layers(i + 1).weights.mmul(priorDerivative)
        } else priorDerivative

        val derivative = layers(i).activation.derivative(outputs(i).compositionOutput.get, outputs(i).activationOutput)

        derivative.mul(priorDerivativeWeighted)
    }
}