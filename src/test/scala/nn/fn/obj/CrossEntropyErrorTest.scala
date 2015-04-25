package nn.fn.obj

import org.jblas.DoubleMatrix
import org.scalatest.{FlatSpec, Matchers, FreeSpec}

class CrossEntropyErrorTest extends FlatSpec with Matchers {
  it should "a = 0.1, y = 1" in {
    val y = new DoubleMatrix(1, 1, 0.1)
    val t = new DoubleMatrix(1, 1, 1)

    CrossEntropyError(y, t) should equal(2.3025850929940455)
  }

  it should "a = 0.9, t = 0" in {
    val y = new DoubleMatrix(1, 1, 0.9)
    val t = new DoubleMatrix(1, 1, 0)

    CrossEntropyError(y, t) should equal(2.302585092994046)
  }

  it should "a = 0.1, y = 0" in {
    val y = new DoubleMatrix(1, 1, 0.01)
    val t = new DoubleMatrix(1, 1, 0)

    CrossEntropyError(y, t) should equal(0.01005033585350145)
  }

  it should "multi" in {
    val y = new DoubleMatrix(2, 1, 0.01, 0.99)
    val t = new DoubleMatrix(2, 1, 0, 1)

    CrossEntropyError(y, t) should equal(0.0201006717070029)
  }
}
