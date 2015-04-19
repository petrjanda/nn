package nn

import nn.fn.act.Logistic
import nn.utils.MatBuilder
import org.jblas.DoubleMatrix

import scala.util.Random

class RBM(val numVisible: Int, val numHidden: Int)(implicit rng: Random) extends Serializable {
  @deprecated
  val W: Array[Array[Double]] = Array.ofDim[Double](numHidden, numVisible)

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

  var wmat = MatBuilder(numHidden, numVisible, W)
  var hbmat = MatBuilder(numHidden, hBias)
  var vbmat = MatBuilder(numVisible, vBias)

  def propagateUpM(v: DoubleMatrix): DoubleMatrix =
    Logistic(wmat.transpose.mmul(v).addColumnVector(hbmat))

  def propagateDownM(v: DoubleMatrix): DoubleMatrix =
    Logistic(wmat.mmul(v).addColumnVector(vbmat))

  def reconstructM(dataSet:DoubleMatrix): DoubleMatrix =
    propagateDownM(propagateUpM(dataSet))

  def updateWeights(diff:(DoubleMatrix, DoubleMatrix, DoubleMatrix)) = {
    wmat = wmat.add(diff._1)
    hbmat = hbmat.add(diff._2)
    vbmat = vbmat.add(diff._3)
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
