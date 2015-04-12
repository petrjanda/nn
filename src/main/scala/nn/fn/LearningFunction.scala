package nn.fn

trait LearningFunction {
  def apply(iteration: Int): Double
}