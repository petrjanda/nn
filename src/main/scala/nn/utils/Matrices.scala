package nn.utils

import org.jblas.DoubleMatrix

object Matrices {
  implicit def matrixToRichMatrix(m: DoubleMatrix) =
    new RichMatrix(m)

  class RichMatrix(m: DoubleMatrix) {
    def round(precision: Int = 0): DoubleMatrix = {
      val exp = math.pow(10, precision)

      new DoubleMatrix(m.rows, m.columns, m.data.map(i => (i * exp).round / exp.toDouble  ):_*)
    }

    def filterInvaid: DoubleMatrix = {
      new DoubleMatrix(m.rows, m.columns, m.data.map {
        case x if x.isNaN => 0.0
        case x => x
      }:_*)
    }
  }
}
