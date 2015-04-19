package nn.trainers.gibbs

import org.jblas.DoubleMatrix

case class GibbsHVHSample(vhMean:DoubleMatrix, vhSample:DoubleMatrix, hvMean:DoubleMatrix, hvSample:DoubleMatrix)

object GibbsHVHSample {
  def apply(vh:GibbsSample, hv:GibbsSample): GibbsHVHSample = {
    GibbsHVHSample(vh.mean, vh.sample, hv.mean, hv.sample)
  }
}