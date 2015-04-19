package nn.trainers

import nn.ds.DataSet
import nn.trainers.gibbs.{GibbsHVHSample, GibbsSample, GibbsSampler}
import nn.{Fn, RBM}
import org.jblas.DoubleMatrix

import scala.util.Random

case class ContrastiveDivergenceTrainer(nn:RBM, iterations:Int, miniBatchSize:Int, numParallel:Int, learningRate:Double, k:Int)(implicit rng:Random) {
  import scala.collection.JavaConversions._

  def train(dataSet:DataSet) = {
    dataSet.miniBatches(miniBatchSize).grouped(numParallel).take(iterations).zipWithIndex.foreach {
      case (batches, iteration) =>
        val batch = batches(0)

        println("Iteration:%5d".format(iteration + 1))

        batch.inputs.columnsAsList.toList.foreach { item =>
          contrastiveDivergence(batch.numExamples, item)
        }
    }
  }

  def train(inputs:DoubleMatrix) = {
    0.until(iterations).foreach { _ =>
      inputs.columnsAsList.toList.foreach { item =>
        nn.updateWeights(
          contrastiveDivergence(inputs.columns, item)
        )
      }
    }
  }

  def contrastiveDivergence(inputLength:Int, input: DoubleMatrix) = {
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

    calculateDiff(inputSample, input, g, inputLength)
  }

  def calculateDiff(inputSample: GibbsSample, input: DoubleMatrix, g: GibbsHVHSample, inputLength: Int) = {
    val weights = inputSample.mean
      .mmul(input.transpose)
      .sub(g.hvMean.mmul(g.vhSample.transpose))
      .mul(learningRate / inputLength)
      .transpose

    val hBias = inputSample.sample.sub(g.hvMean).mul(learningRate / inputLength)

    val vBias = input.sub(g.vhSample).mul(learningRate / inputLength)

    (weights, hBias, vBias)
  }
}







