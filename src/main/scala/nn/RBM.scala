package nn

import nn.ds.DataSet
import nn.fn.ObjectiveFunction
import nn.fn.act.Logistic
import nn.utils.{Fn, MatBuilder}
import org.jblas.DoubleMatrix

import scala.util.Random

object RBM {
  def apply(numVisible: Int, numHidden: Int, objective:ObjectiveFunction)(implicit rng: Random) = {
    val w = Fn.uniformMatrix(numVisible, numHidden, 1 / numVisible)
    val h = new DoubleMatrix(1, numHidden).fill(0.0)
    val v = new DoubleMatrix(1, numVisible).fill(0.0)

    new RBM(w, h, v, objective)
  }
}

class RBM(val w:DoubleMatrix, val h:DoubleMatrix, val v:DoubleMatrix, objective:ObjectiveFunction) extends Serializable {
  def propagateUp(value: DoubleMatrix): DoubleMatrix =
    Logistic(w.transpose.mmul(value).addColumnVector(h))

  def propagateDown(value: DoubleMatrix): DoubleMatrix =
    Logistic(w.mmul(value).addColumnVector(v))

  def reconstruct(dataSet: DataSet): DoubleMatrix =
    propagateDown(propagateUp(dataSet.features))

  def loss(data: DataSet): Double = {
    val outputs = reconstruct(data)
    
    objective(outputs, data.features)
  }

  def updateWeights(diff:(DoubleMatrix, DoubleMatrix, DoubleMatrix)):RBM =
    new RBM(w.add(diff._1), h.add(diff._2), v.add(diff._3), objective)
}
