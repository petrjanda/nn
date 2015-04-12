package nn.ds

import org.jblas.DoubleMatrix

abstract trait DataSet {

  val inputs: DoubleMatrix
  val targets: DoubleMatrix

  val numExamples = inputs.columns
  val numInputs = inputs.rows
  val numOutputs = targets.rows

  def copy(inputs: DoubleMatrix, targets: DoubleMatrix): DataSet

  def batch(columns: Array[Int] ): DataSet = {
    copy(inputs.getColumns(columns), targets.getColumns(columns))
  }

  def miniBatches(batchSize: Int): Iterator[DataSet] = {
    Stream.continually((0 until numExamples)).flatten.grouped(batchSize).map {
      columns =>
        batch(columns.toArray)
    }
  }
}
