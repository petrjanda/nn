package nn.fn

import org.jblas.DoubleMatrix

trait ActivationFunction extends DifferentiableFunction {
  def apply(x: DoubleMatrix): DoubleMatrix
}