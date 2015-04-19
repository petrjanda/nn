package nn.trainers.gibbs

import nn.{Fn, RBM}
import org.jblas.DoubleMatrix

import scala.util.Random

class GibbsSampler(rbm:RBM)(implicit rng:Random) {
  val numHidden = rbm.numHidden
  val numVisible = rbm.numVisible

  def sampleGibbsHVH(h0Sample: DoubleMatrix): GibbsHVHSample = {
    val vh = sampleVGivenH(h0Sample)

    GibbsHVHSample(vh, sampleHGivenV(vh.sample))
  }

  def sampleHGivenV(v0Sample: DoubleMatrix): GibbsSample = {
    val mean = rbm.propagateUpM(v0Sample)

    GibbsSample(mean, sample(mean, rng))
  }

  def sampleVGivenH(h0Sample: DoubleMatrix): GibbsSample = {
    val mean = rbm.propagateDownM(h0Sample)

    GibbsSample(mean, sample(mean, rng))
  }

  private def sample(m:DoubleMatrix, rng:Random): DoubleMatrix = {
    new DoubleMatrix(m.rows, m.columns, m.data.map { s => Fn.binomial(1, s, rng) }:_*)
  }
}
