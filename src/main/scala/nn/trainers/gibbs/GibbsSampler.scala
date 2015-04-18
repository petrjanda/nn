package nn.trainers.gibbs

import nn.{Fn, RBM}

import scala.util.Random

class GibbsSampler(rbm:RBM)(implicit rng:Random) {
  val numHidden = rbm.numHidden
  val numVisible = rbm.numVisible

  def sampleGibbsHVH(h0Sample: Array[Double], nvMeans: Array[Double], nvSamples: Array[Double]): GibbsHVHSample = {
    val vh = sampleVGivenH(h0Sample)

    GibbsHVHSample(vh, sampleHGivenV(vh.sample))
  }

  def sampleHGivenV(v0Sample: Array[Double]): GibbsSample = {
    val mean = Range(0, numHidden).map { i => rbm.propagateUp(v0Sample, i) }.toArray

    GibbsSample(mean, sample(mean, rng))
  }

  def sampleVGivenH(h0Sample: Array[Double]): GibbsSample = {
    val mean = Range(0, numVisible).map { i => rbm.propagateDown(h0Sample, i) }.toArray

    GibbsSample(mean, sample(mean, rng))
  }

  private def sample(m:Array[Double], rng:Random) = {
    m.map { s => Fn.binomial(1, s, rng) }
  }
}
