package nn

import scala.util.Random

class RBM(val numVisible: Int, val numHidden: Int)(implicit rng: Random) {
  val a: Double = 1 / numVisible
  var W: Array[Array[Double]] = Array.ofDim[Double](numHidden, numVisible)
  var hBias: Array[Double] = Array.fill(numHidden) { 0.0 }
  var vBias: Array[Double] = Array.fill(numVisible) { 0.0 }

  Range(0, numHidden).foreach { i =>
    Range(0, numVisible).foreach { j =>
      W(i)(j) = Fn.uniform(-a, a, rng)
    }
  }

  def propagateUp(v: Array[Int], i: Int): Double = {
    val w = W(i)
    val b = hBias(i)

    Fn.sigmoid(
      Range(0, numVisible).toArray.foldLeft(0.0) { (t, j) => t + w(j) * v(j) } + b
    )
  }

  def propagateDown(h: Array[Int], i: Int): Double = {
    Fn.sigmoid(
      Range(0, numHidden).toArray.foldLeft(0.0) { (t, j) => t + W(j)(i) * h(j) } + vBias(i)
    )
  }

  def reconstruct(v: Array[Array[Int]]): Array[Array[Double]] = {
    v.map { v =>
      val h = Range(0, numHidden).toArray.map { i =>
        propagateUp(v, i)
      }

      val layer = Layer(numHidden, vBias, W, h)

      Range(0, numVisible).toArray.map { layer.activationOutput(_) }
    }
  }

  case class Layer(n_hidden: Int, vbias:Array[Double], W: Array[Array[Double]], h: Array[Double]) {
    def activationOutput(i: Int) = {
      Fn.sigmoid(
        0.until(n_hidden).foldLeft(0.0) { (t, j) => t + W(j)(i) * h(j) } + vbias(i)
      )
    }
  }
}

object Fn {
  def uniform(min: Double, max: Double, rng:Random): Double = rng.nextDouble() * (max - min) + min
  def binomial(n: Int, p: Double, rng:Random): Int = {
    if(p < 0 || p > 1) return 0

    var c: Int = 0
    var r: Double = 0

    var i: Int = 0
    for(i <- 0 until n) {
      r = rng.nextDouble()
      if(r < p) c += 1
    }

    c
  }

  def sigmoid(x: Double): Double = 1.0 / (1.0 + math.pow(math.E, -x))
}
