package nn

import org.jblas.DoubleMatrix

case class LayerState(compositionOutput: Option[DoubleMatrix], activationOutput: DoubleMatrix)
