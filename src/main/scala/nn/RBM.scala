package nn

import nn.fn.act.Logistic
import nn.utils.MatBuilder
import org.jblas.DoubleMatrix

import scala.util.Random

object RBM {
  def apply(numVisible: Int, numHidden: Int)(implicit rng: Random) = {
    val W = Fn.uniformMatrix(numVisible, numHidden, 1 / numVisible)
    val hBias = new DoubleMatrix(1, numHidden).fill(0.0)
    val vBias = new DoubleMatrix(1, numVisible).fill(0.0)

    new RBM(numVisible, numHidden, W, hBias, vBias)
  }
}

class RBM(val numVisible: Int, val numHidden: Int, W:DoubleMatrix, hBias:DoubleMatrix, vBias:DoubleMatrix)(implicit rng: Random) extends Serializable {
  def propagateUpM(v: DoubleMatrix): DoubleMatrix =
    Logistic(W.transpose.mmul(v).addColumnVector(hBias))

  def propagateDownM(v: DoubleMatrix): DoubleMatrix =
    Logistic(W.mmul(v).addColumnVector(vBias))

  def reconstructM(dataSet:DoubleMatrix): DoubleMatrix =
    propagateDownM(propagateUpM(dataSet))

  def updateWeights(diff:(DoubleMatrix, DoubleMatrix, DoubleMatrix)):RBM = {
    new RBM(numVisible, numHidden, W.add(diff._1), hBias.add(diff._2), vBias.add(diff._3))
  }
}

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
