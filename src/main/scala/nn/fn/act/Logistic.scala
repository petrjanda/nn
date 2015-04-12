package nn.fn.act

import nn.fn.ActivationFunction
import org.jblas.DoubleMatrix
import org.jblas.MatrixFunctions._

object Logistic extends ActivationFunction {
  def apply(x: DoubleMatrix): DoubleMatrix =
    powi(expi(x.neg).addi(1), -1)

  def derivative(x: DoubleMatrix, y: DoubleMatrix): DoubleMatrix =
    y.mul(y.neg.addi(1))
}
