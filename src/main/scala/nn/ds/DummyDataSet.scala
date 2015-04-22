package nn.ds

import org.jblas.DoubleMatrix

case class DummyDataSet(features: DoubleMatrix, targets: DoubleMatrix) extends DataSet {
  def cp(inputs: DoubleMatrix, targets: DoubleMatrix): DataSet =
    DummyDataSet(inputs, targets)
}
