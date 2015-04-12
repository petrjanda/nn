package nn

import org.jblas.DoubleMatrix

case class Scores(results:DoubleMatrix, targets:DoubleMatrix) {
  def scores = results.columnArgmaxs.zip(targets.columnArgmaxs).map {
    case (p, t) if p == t => 1
    case _ => 0
  }

  def average: Double =
    scores.sum.toDouble / scores.length
}
