package nn.fn.scr

import nn.fn.ScoreFunction
import org.jblas.DoubleMatrix
import org.jblas.MatrixFunctions._

case class BinaryClassificationScore(threshold:Double) extends ScoreFunction {
  def score(results:DoubleMatrix, targets:DoubleMatrix): Double = {
    val scores = abs(targets.sub(floor(results.add(threshold))))

    1.0 - (scores.sum.abs / results.length)
  }
}
