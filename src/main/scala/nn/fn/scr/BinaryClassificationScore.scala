package nn.fn.scr

import nn.fn.ScoreFunction
import org.jblas.DoubleMatrix
import org.jblas.MatrixFunctions._

case class BinaryClassificationScore(threshold:Double) extends ScoreFunction {
  def score(results:DoubleMatrix, targets:DoubleMatrix): Double = {
    val diffs = abs(targets.sub(floor(results.add(threshold))))

    diffs.sum.abs / results.columns
  }
}

case object AbsoluteDiffScore extends ScoreFunction {
  def score(results:DoubleMatrix, targets:DoubleMatrix): Double = {
    val diffs = abs(targets.sub(results))

    diffs.sum.abs / results.columns
  }
}
