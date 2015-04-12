package nn.fn.lrn

import nn.fn.LearningFunction

object ConstantRate {
  def apply(rate: Double): ConstantRate = new ConstantRate(rate)
}

class ConstantRate(rate: Double) extends LearningFunction {
  def apply(iteration: Int): Double = rate
}