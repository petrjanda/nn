package nn.fn.act

import nn.fn.ActivationFunction
import org.jblas.DoubleMatrix
import org.jblas.MatrixFunctions._

object SoftMax extends ActivationFunction {
  def apply(x: DoubleMatrix): DoubleMatrix = {
    val normalizer = logSumExp(x)
    val logProb = x.subi(normalizer.repmat(x.rows, 1))
    exp(logProb)
  }

  def logSumExp(input: DoubleMatrix): DoubleMatrix = {
    val maxSmall = input.columnMaxs
    val maxBig = maxSmall.repmat(input.rows, 1)
    val l = logi(expi(input.sub(maxBig)).columnSums)
    l.addi(maxSmall)
  }

  def derivative(x: DoubleMatrix, y: DoubleMatrix): DoubleMatrix =
    y.mul(0).addi(1)
}
