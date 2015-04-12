package nn

class LayerList(val layers: List[Layer]) {
  def :+(partial: PartialLayer): LayerList = new LayerList(layers:+Layer(layers.last.numOutputs, partial.numOutputs, partial.activation))
}
