package nn.trainers

import nn.trainers.gibbs.{GibbsHVHSample, GibbsSample, GibbsSampler}
import nn.{Fn, RBM}
import org.jblas.DoubleMatrix

import scala.util.Random

case class ContrastiveDivergenceTrainer(nn:RBM, iterations:Int, learningRate:Double, k:Int)(implicit rng:Random) {
  def train(dataSet:DoubleMatrix) = {
    import scala.collection.JavaConversions._

    0.until(iterations).foreach { _ =>
      dataSet.columnsAsList.toList.foreach { item =>
        contrastiveDivergence(dataSet.length, nn, item.toArray, learningRate, k)
      }
    }
  }

  def contrastiveDivergence(inputLength:Int, rbm:RBM, input: Array[Double], lr: Double, k: Int) {
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

    updateRbm(rbm, inputSample, input, g, inputLength, lr)
  }
  
  def updateRbm(rbm: RBM, inputSample: GibbsSample, input: Array[Double], g: GibbsHVHSample, inputLength: Int, lr: Double) = {
    val numHidden = rbm.numHidden
    val numVisible = rbm.numVisible

    // Update weights and bias
    Range(0, numHidden).foreach { i =>
      Range(0, numVisible).foreach { j =>
        rbm.W(i)(j) += lr * (inputSample.mean(i) * input(j) - g.hvMean(i) * g.vhSample(j)) / inputLength
      }

      rbm.hBias(i) += lr * (inputSample.sample(i) - g.hvMean(i)) / inputLength
    }

    Range(0, numVisible).foreach { i =>
      rbm.vBias(i) += lr * (input(i) - g.vhSample(i)) / inputLength
    }
  }
}







