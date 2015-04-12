package nn.fn

import org.jblas.DoubleMatrix

trait DifferentiableFunction extends Serializable {
  def derivative(x: DoubleMatrix, y: DoubleMatrix): DoubleMatrix
}
