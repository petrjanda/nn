package nn.ds

import org.jblas.DoubleMatrix

case class DummyDataSet(inputs: DoubleMatrix, targets: DoubleMatrix) extends DataSet {
  def copy(inputs: DoubleMatrix, targets: DoubleMatrix): DataSet =
    DummyDataSet(inputs, targets)
}
