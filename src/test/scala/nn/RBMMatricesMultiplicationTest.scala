package nn

import org.jblas.DoubleMatrix
import org.scalatest.{Matchers, FreeSpec}

class RBMMatricesMultiplicationTest extends FreeSpec with Matchers {
  "" - {
    val a = new DoubleMatrix(3, 2, 1, 2, 3, 4, 5, 6)

    val b = new DoubleMatrix(3, 4, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1)

    println(a.transpose.mmul(b).getColumn(0))
  }

}
