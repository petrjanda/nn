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
        contrastiveDivergence(dataSet.length, item)
      }
    }
  }

  def contrastiveDivergence(inputLength:Int, input: DoubleMatrix) {
    val gibbs = new GibbsSampler(nn)
    val numHidden = nn.numHidden
    val numVisible = nn.numVisible

    val inputSample = gibbs.sampleHGivenV(input)

    val first = GibbsHVHSample(
      new DoubleMatrix(1, numVisible).fill(0.0),
      new DoubleMatrix(1, numHidden).fill(0.0),
      inputSample.mean,
      inputSample.sample
    )

    val g = 0.until(k).foldLeft(first) ( (old, _) => {
      gibbs.sampleGibbsHVH(old.hvSample)
    })

    updateRbm(inputSample, input, g, inputLength)
  }
  
  def updateRbm(inputSample: GibbsSample, input: DoubleMatrix, g: GibbsHVHSample, inputLength: Int) = {
    val numHidden = nn.numHidden
    val numVisible = nn.numVisible

    // Update weights and bias
    Range(0, numHidden).foreach { i =>
      Range(0, numVisible).foreach { j =>
        nn.W(i)(j) += learningRate * (
          inputSample.mean.data(i) * input.data(j) -
            g.hvMean.data(i) * g.vhSample.data(j)
          ) / inputLength
      }

      nn.hBias(i) += learningRate * (inputSample.sample.data(i) - g.hvMean.data(i)) / inputLength
    }

    Range(0, numVisible).foreach { i =>
      nn.vBias(i) += learningRate * (input.data(i) - g.vhSample.data(i)) / inputLength
    }
  }
}







