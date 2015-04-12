package nn

import nn.ds.DataSet
import nn.fn.{WeightDecay, ObjectiveFunction}
import org.jblas.DoubleMatrix

case class Derivatives(layers:List[Layer], objective:ObjectiveFunction, weightDecay: WeightDecay) {
  def errorGradients(data: DataSet, outputs:List[LayerState]): Seq[DoubleMatrix] = {

    val errorDerivative = objective.derivative(outputs.last.activationOutput, data.targets)

    derivatives(errorDerivative, layers, outputs).zipWithIndex.map {
      case (derivative, i) =>
        val gradient = if (i > 0) {
          outputs(i - 1).activationOutput.mmul(derivative.transpose)
        } else {
          data.inputs.mmul(derivative.transpose)
        }

        if(i < layers.size) {
          weightDecay.derivative(layers(i), gradient)
        } else gradient
    }
  }

  private def derivatives(initial:DoubleMatrix, layers:List[Layer], outputs:List[LayerState]) =
    (0 until layers.size).scanRight(initial) {
      case (i, priorDerivative) =>
        val priorDerivativeWeighted = if (i < layers.size - 1) {
          layers(i + 1).weights.mmul(priorDerivative)
        } else priorDerivative
        val derivative = layers(i).activation.derivative(outputs(i).compositionOutput.get, outputs(i).activationOutput)
        derivative.mul(priorDerivativeWeighted)
    }
}
