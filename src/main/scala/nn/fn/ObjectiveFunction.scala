package nn.fn

import org.jblas.DoubleMatrix

trait ObjectiveFunction extends DifferentiableFunction {
  def apply(y: DoubleMatrix, t: DoubleMatrix): Double
}
