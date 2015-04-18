package nn.utils

import org.jblas.DoubleMatrix

object MatBuilder {
  def apply(rows:Int, columns:Int, a:Array[Array[Double]]) =
    new DoubleMatrix(columns, rows, a.flatten:_*)

  def apply(rows:Int, columns:Int, a:Array[Array[Int]]) =
    new DoubleMatrix(columns, rows, a.flatten.map(_.toDouble):_*)

  def apply(rows:Int, columns:Int, fn:() => Double) =
    new DoubleMatrix(columns, rows, Range(0, columns * rows).map(_ => fn()):_*)

  def apply(columns:Int, a:Array[Double]) =
    new DoubleMatrix(1, columns, a:_*)
}
