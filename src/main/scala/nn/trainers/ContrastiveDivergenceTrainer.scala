package nn.trainers

import nn.{Fn, RBM}

import scala.util.Random

case class ContrastiveDivergenceTrainer(iterations:Int, learningRate:Double, k:Int) {
  def train(rbm:RBM, dataSet:Array[Array[Int]]) = {
    0.until(iterations).foreach { _ =>
      dataSet.foreach { item =>
        contrastiveDivergence(rbm, item, learningRate, k)
      }
    }
  }

  def contrastiveDivergence(rbm:RBM, input: Array[Int], lr: Double, k: Int) {
    val gibbs = new GibbsSampler(rbm)
    val numHidden = rbm.numHidden
    val numVisible = rbm.numVisible

    val inputSample = gibbs.sampleHGivenV(input)
    val first = GibbsHVHSample(
      Array.fill(numVisible) { 0.0 },
      Array.fill(numHidden) { 0 },
      inputSample.mean,
      inputSample.sample
    )

    val g = 0.until(k).foldLeft(first) ( (old, _) => {
      gibbs.sampleGibbsHVH(old.hvSample, old.vhMean, old.vhSample)
    })

    // Update weights and bias
    Range(0, numHidden).foreach { i =>
      Range(0, numVisible).foreach { j =>
        rbm.W(i)(j) += lr * (inputSample.mean(i) * input(j) - g.hvMean(i) * g.vhSample(j)) / rbm.N
      }

      rbm.hBias(i) += lr * (inputSample.sample(i) - g.hvMean(i)) / rbm.N
    }

    Range(0, numVisible).foreach { i =>
      rbm.vBias(i) += lr * (input(i) - g.vhSample(i)) / rbm.N
    }
  }
}


object GibbsHVHSample {
  def apply(vh:GibbsSample, hv:GibbsSample): GibbsHVHSample = {
    GibbsHVHSample(vh.mean, vh.sample, hv.mean, hv.sample)
  }
}

case class GibbsHVHSample(vhMean:Array[Double], vhSample:Array[Int], hvMean:Array[Double], hvSample:Array[Int])

case class GibbsSample(mean:Array[Double], sample:Array[Int])

class GibbsSampler(rbm:RBM) {
  val rng = rbm.rng
  val numHidden = rbm.numHidden
  val numVisible = rbm.numVisible

  def sampleGibbsHVH(h0Sample: Array[Int], nvMeans: Array[Double], nvSamples: Array[Int]): GibbsHVHSample = {
    val vh = sampleVGivenH(h0Sample)

    GibbsHVHSample(vh, sampleHGivenV(vh.sample))
  }

  def sampleHGivenV(v0Sample: Array[Int]): GibbsSample = {
    val mean = Range(0, numHidden).map { i => rbm.propagateUp(v0Sample, i) }.toArray

    GibbsSample(mean, sample(mean, rng))
  }

  def sampleVGivenH(h0Sample: Array[Int]): GibbsSample = {
    val mean = Range(0, numVisible).map { i => rbm.propagateDown(h0Sample, i) }.toArray

    GibbsSample(mean, sample(mean, rng))
  }

  private def sample(m:Array[Double], rng:Random) = {
    m.map { s => Fn.binomial(1, s, rng) }
  }
}