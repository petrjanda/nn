package nn.fn

import org.jblas.DoubleMatrix

trait ScoreFunction {
  def score(results:DoubleMatrix, targets:DoubleMatrix): Double
}
