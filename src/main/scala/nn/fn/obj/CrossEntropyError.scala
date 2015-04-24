package nn.fn.obj

import nn.fn.ObjectiveFunction
import org.jblas.DoubleMatrix
import org.jblas.MatrixFunctions._

object CrossEntropyError extends ObjectiveFunction {
  def apply(a: DoubleMatrix, y: DoubleMatrix): Double = {
    import nn.utils.Matrices._

    - y.mul(log(a)).add(y.neg.add(1).mul(log(a.neg.add(1)))).columnSums.filterInvaid.mean
  }

  def derivative(x: DoubleMatrix, y: DoubleMatrix): DoubleMatrix =
    x.sub(y).muli(1.0 / y.columns)
}