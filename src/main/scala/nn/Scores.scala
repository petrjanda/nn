package nn

import org.jblas.DoubleMatrix
import org.jblas.MatrixFunctions._

case class Scores(results:DoubleMatrix, targets:DoubleMatrix) {
  val scores = targets.sub(floor(results.add(.5)))

  def average: Double =
    1.0 - (scores.sum.abs / results.length)
}
