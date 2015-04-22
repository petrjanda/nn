package nn.trainers.gibbs

import nn.RBM
import nn.utils.Fn
import org.jblas.DoubleMatrix

import scala.util.Random

class GibbsSampler(rbm:RBM)(implicit rng:Random) {
  def sampleGibbsHVH(h: DoubleMatrix): GibbsHVHSample = {
    val vh = sampleVGivenH(h)

    GibbsHVHSample(vh, sampleHGivenV(vh.sample))
  }

  def sampleHGivenV(v: DoubleMatrix): GibbsSample = {
    val mean = rbm.propagateUp(v)

    GibbsSample(mean, sample(mean))
  }

  def sampleVGivenH(h: DoubleMatrix): GibbsSample = {
    val mean = rbm.propagateDown(h)

    GibbsSample(mean, sample(mean))
  }

  private def sample(m:DoubleMatrix): DoubleMatrix = {
    new DoubleMatrix(m.rows, m.columns, m.data.map { s => Fn.binomial(1, s) }:_*)
  }
}
