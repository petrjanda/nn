package nn.fn.act

import nn.fn.ActivationFunction
import org.jblas.DoubleMatrix
import org.jblas.MatrixFunctions._

object HyperbolicTangent extends ActivationFunction {
  def apply(x: DoubleMatrix): DoubleMatrix =
    tanh(x)

  def derivative(x: DoubleMatrix, y: DoubleMatrix): DoubleMatrix =
    pow(y,2).negi.addi(1)
}
