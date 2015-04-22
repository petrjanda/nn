package nn.ds

import org.jblas.DoubleMatrix

abstract trait DataSet {

  def features: DoubleMatrix
  def targets: DoubleMatrix

  lazy val numExamples = features.columns
  lazy val numInputs = features.rows
  lazy val numOutputs = targets.rows

  def cp(inputs: DoubleMatrix, targets: DoubleMatrix): DataSet

  def batch(columns: Array[Int] ): DataSet = {
    cp(features.getColumns(columns), targets.getColumns(Array(0)))
  }

  def miniBatches(batchSize: Int): Iterator[DataSet] = {
    Stream.continually((0 until numExamples)).flatten.grouped(batchSize).map {
      columns =>
        batch(columns.toArray)
    }
  }
}