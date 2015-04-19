package nn.trainers

import nn.ds.DataSet
import nn.trainers.gibbs.{GibbsHVHSample, GibbsSample, GibbsSampler}
import nn.{Fn, RBM}
import org.jblas.DoubleMatrix

import scala.util.Random

case class ContrastiveDivergenceTrainer(nn:RBM, iterations:Int, learningRate:Double, k:Int)(implicit rng:Random) {
  import scala.collection.JavaConversions._

  def train(dataSet:DataSet) = {
    dataSet.miniBatches(1000).grouped(1).take(iterations).zipWithIndex.foreach {
      case (batches, iteration) =>
        val batch = batches(0)

        println("Iteration:%5d".format(iteration + 1))

        batch.inputs.columnsAsList.toList.foreach { item =>
          contrastiveDivergence(batch.numExamples, item)
        }
    }
  }

  def train(inputs:DoubleMatrix) = {
    inputs.columnsAsList.toList.foreach { item =>
      contrastiveDivergence(inputs.columns, item)
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







