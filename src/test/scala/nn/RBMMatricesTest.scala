package nn

import org.jblas.DoubleMatrix
import org.scalatest.{Matchers, FreeSpec}

class RBMMatricesTest extends FreeSpec with Matchers {
  "from nested list" - {
    val a = Array.ofDim[Double](2, 3)

    a(0)(0) = 1.0
    a(0)(1) = 2.0
    a(0)(2) = 3.0

    a(1)(0) = 11.0
    a(1)(1) = 12.0
    a(1)(2) = 13.0

    val m = MatBuilder(2, 3, a)

    m.getColumn(0).toArray.toList should equal(List(1.0, 2.0, 3.0))
  }

  "from list" - {
    val a = Array.ofDim[Double](2)

    a(0) = 1.0
    a(1) = 2.0

    val m = MatBuilder(2, a)

    m.getRow(0).toArray.toList should equal(List(1.0, 2.0))
  }
}
