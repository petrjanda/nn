package nn.utils

import org.jblas.DoubleMatrix

import scala.util.Random

/**
 * TODO Refactor
 */
object Fn {
  def uniformMatrix(r:Int, c:Int, a:Double)(implicit rng:Random) = new DoubleMatrix(r, c, Range(0, r* c).map(_ => Fn.uniform(-a, a, rng)):_*)

  def uniform(min: Double, max: Double, rng:Random): Double = rng.nextDouble() * (max - min) + min

  def binomial(n: Int, p: Double, rng:Random): Double = {
    if(p < 0 || p > 1) return 0

    var c: Int = 0
    var r: Double = 0

    var i: Int = 0
    for(i <- 0 until n) {
      r = rng.nextDouble()
      if(r < p) c += 1
    }

    c.toDouble
  }
}
