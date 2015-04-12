package nn


import nn.ds.DataSet
import nn.fn.{WeightDecay, ObjectiveFunction}
import org.jblas.DoubleMatrix

class NeuralNetwork(val layers: List[Layer], val objective: ObjectiveFunction, val weightDecay: WeightDecay) extends Serializable {
  private val derivatives = new Derivatives(layers, objective, weightDecay)

  def weights: List[DoubleMatrix] = layers.map(_.weights)

  def momentum: List[DoubleMatrix] = weights.map(_.mul(0))

  def eval(data: DataSet): Double = {
    val outputs = compute(data.inputs)

    Scores(outputs, data.targets).average
  }

  def copy(layers: List[Layer]): NeuralNetwork =
    new NeuralNetwork(layers, objective, weightDecay)

  def errorGradients(data: DataSet): Seq[DoubleMatrix] =
    derivatives.errorGradients(data, propagate(data.inputs))

  def loss(data: DataSet): Double = {
    val outputs = propagate(data.inputs)
    weightDecay(layers, objective(outputs.last.activationOutput, data.targets))
  }

  def compute(inputs: DoubleMatrix): DoubleMatrix =
    propagate(inputs).last.activationOutput

  private def propagate(inputs: DoubleMatrix): List[LayerState] =
    layers.scanLeft(LayerState(None, inputs)) {
      case (LayerState(_, x), layer) => layer(x)
    }.tail
}

object NeuralNetwork {
  def apply(layers: List[Layer], objective: ObjectiveFunction, weightDecay: WeightDecay = WeightDecay.ZERO): NeuralNetwork =
    new NeuralNetwork(layers, objective, weightDecay)
}