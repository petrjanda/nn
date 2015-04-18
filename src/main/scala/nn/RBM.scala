package nn

import nn.fn.act.Logistic
import nn.utils.MatBuilder
import org.jblas.DoubleMatrix

import scala.util.Random

class RBM(val numVisible: Int, val numHidden: Int)(implicit rng: Random) {
  @deprecated
  var W: Array[Array[Double]] = Array.ofDim[Double](numHidden, numVisible)

  @deprecated
  var hBias: Array[Double] = Array.fill(numHidden) { 0.0 }

  @deprecated
  var vBias: Array[Double] = Array.fill(numVisible) { 0.0 }

  val a: Double = 1 / numVisible
  Range(0, numHidden).foreach { i =>
    Range(0, numVisible).foreach { j =>
      W(i)(j) = Fn.uniform(-a, a, rng)
    }
  }

  def wmat = MatBuilder(numHidden, numVisible, W)
  def hbmat = MatBuilder(numHidden, hBias)
  def vbmat = MatBuilder(numVisible, vBias)

  def propagateUpM(v: DoubleMatrix): DoubleMatrix = {
    Logistic(wmat.transpose.mmul(v).addColumnVector(hbmat))
  }

  def propagateDownM(v: DoubleMatrix): DoubleMatrix = {
    Logistic(wmat.mmul(v).addColumnVector(vbmat))
  }

  def reconstructM(dataSet:DoubleMatrix): Array[DoubleMatrix] = {
    import scala.collection.JavaConversions._

    val h = propagateUpM(dataSet).transpose.rowsAsList()

    h.map { col =>
      Layer(vbmat, wmat, col).activationOutput
    }.toArray
  }

  case class Layer(vbias: DoubleMatrix, W: DoubleMatrix, h: DoubleMatrix) {
    def activationOutput: DoubleMatrix =
      Logistic(W.mmul(h.transpose).addColumnVector(vbias))
  }



  @deprecated
  def propagateDown(h: Array[Double], i: Int): Double = {
    val b = vBias(i)
    Fn.sigmoid(
      Range(0, numHidden).toArray.foldLeft(0.0) { (t, j) => t + W(j)(i) * h(j) } + b
    )
  }

  @deprecated
  def propagateUp(v: Array[Double], i: Int): Double = {
    val w = W(i)
    val b = hBias(i)

    Fn.sigmoid(
      Range(0, numVisible).toArray.foldLeft(0.0) { (t, j) => t + w(j) * v(j) } + b
    )
  }

  @deprecated
  def reconstruct(v: Array[Array[Double]]): Array[Array[Double]] = {
    v.map { v =>
      val h = Range(0, numHidden).toArray.map { i =>
        propagateUp(v, i)
      }

      val layer = Layer(vbmat, wmat, MatBuilder(numHidden, h))

      layer.activationOutput.toArray
    }
  }
}

object Fn {
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

  @deprecated
  def sigmoid(x: Double): Double = 1.0 / (1.0 + math.pow(math.E, -x))
}
