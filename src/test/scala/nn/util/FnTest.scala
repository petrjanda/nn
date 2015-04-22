package nn.util

import nn.utils.Fn
import org.scalatest.{FreeSpec, Matchers}

import scala.util.Random

class FnTest extends FreeSpec with Matchers {
  "binomial distribution for p = 1 should be n" - {
    implicit val rng = new Random(1234)

    Fn.binomial(3, 1) should equal(3.0)
  }

}

