package nn.fn.lrn

import nn.fn.LearningFunction

object AnnealingRate {
  def apply(rate: Double, iterations: Int): AnnealingRate = new AnnealingRate(rate, iterations: Int)
}

class AnnealingRate(rate: Double, iterations: Int) extends LearningFunction {
  def apply(iteration: Int): Double = rate / (1 + iteration.toDouble / iterations)
}