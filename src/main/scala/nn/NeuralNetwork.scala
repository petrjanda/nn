package nn

import nn.ds.DataSet
import nn.fn.{ScoreFunction, WeightDecay, ObjectiveFunction}
import org.jblas.DoubleMatrix

class NeuralNetwork(val layers: List[Layer], val objective: ObjectiveFunction, val score: ScoreFunction, val weightDecay: WeightDecay) extends Serializable {
  def weights: List[DoubleMatrix] = layers.map(_.weights)

  def momentums: List[DoubleMatrix] = weights.map(_.mul(0))

  def eval(data: DataSet): Double =
    score.score(compute(data.features), data.targets)

  def loss(data: DataSet): Double = {
    val outputs = propagate(data.features)
    weightDecay(layers, objective(outputs.last.activationOutput, data.targets))
  }

  def compute(inputs: DoubleMatrix): DoubleMatrix =
    propagate(inputs).last.activationOutput

  /**
   * Propagate given inputs throughout the network and calculate layer state as the output (composition and activation).
   *
   * @param inputs
   * @return
   */
  def propagate(inputs: DoubleMatrix): List[LayerState] =
    layers.scanLeft(LayerState(None, inputs)) {
      case (LayerState(_, x), layer) => layer(x)
    }.tail

  /**
   * Create a new Neural Network as a copy of this one.
   *
   * @param layers New list of layers.
   * @return Neural Network.
   */
  def copy(layers: List[Layer]): NeuralNetwork =
    NeuralNetwork(layers, objective, score, weightDecay)
}

object NeuralNetwork {
  def apply(layers: List[Layer], objective: ObjectiveFunction, score: ScoreFunction, weightDecay: WeightDecay = WeightDecay.ZERO): NeuralNetwork =
    new NeuralNetwork(layers, objective, score, weightDecay)
}