package nn.fn.act

import org.jblas.DoubleMatrix
import org.scalatest.{Matchers, FlatSpec}

class HyperbolicTangentTest extends FlatSpec with Matchers {
  it should "compute" in {
    val m = new DoubleMatrix(1, 1, 1)

    HyperbolicTangent(m) should equal(new DoubleMatrix(1, 1, 0.7615941559557649))

  }

}
