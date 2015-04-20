package nn.trainers.gibbs

import nn.{Fn, RBM}
import org.jblas.DoubleMatrix

import scala.util.Random

class GibbsSampler(rbm:RBM)(implicit rng:Random) {
  def sampleGibbsHVH(h: DoubleMatrix): GibbsHVHSample = {
    val vh = sampleVGivenH(h)

    GibbsHVHSample(vh, sampleHGivenV(vh.sample))
  }

  def sampleHGivenV(v: DoubleMatrix): GibbsSample = {
    val mean = rbm.propagateUpM(v)

    GibbsSample(mean, sample(mean))
  }

  def sampleVGivenH(h0Sample: DoubleMatrix): GibbsSample = {
    val mean = rbm.propagateDownM(h0Sample)

    GibbsSample(mean, sample(mean))
  }

  private def sample(m:DoubleMatrix): DoubleMatrix = {
    new DoubleMatrix(m.rows, m.columns, m.data.map { s => Fn.binomial(1, s, rng) }:_*)
  }
}
