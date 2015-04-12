package nn.fn.obj

import nn.fn.ObjectiveFunction
import org.jblas.DoubleMatrix
import org.jblas.MatrixFunctions._

object CrossEntropyError extends ObjectiveFunction {
  def apply(y: DoubleMatrix, t: DoubleMatrix): Double =
    -((log(y).muli(t).columnSums).mean)

  def derivative(x: DoubleMatrix, y: DoubleMatrix): DoubleMatrix =
    x.sub(y).muli(1.0 / y.columns)
}
