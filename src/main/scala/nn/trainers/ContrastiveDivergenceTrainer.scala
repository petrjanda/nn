package nn.trainers

import nn.ds.DataSet
import nn.fn.LearningFunction
import nn.trainers.gibbs.{GibbsHVHSample, GibbsSample, GibbsSampler}
import nn.{FeedForwardNN$, RBM}
import org.jblas.DoubleMatrix

import scala.util.Random

case class ContrastiveDivergenceTrainer(private var nn:RBM, iterations:Int, evalIterations:Int, miniBatchSize:Int, numParallel:Int, learningRate:LearningFunction, k:Int)(implicit rng:Random) {
  import scala.collection.JavaConversions._

  def evalIteration(iteration: Int, trainingSet: DataSet) {
    if ((iteration + 1) % evalIterations == 0) {
      val loss = nn.loss(trainingSet)
      println("Iteration:%5d, Loss: %.5f".format(iteration + 1, loss))
    }
  }

  def train(dataSet:DataSet):RBM = {
    dataSet.miniBatches(miniBatchSize).grouped(numParallel).take(iterations).zipWithIndex.foreach {
      case (batches, iteration) =>
        val batch = batches(0)

        evalIteration(iteration, dataSet)

        batch.features.columnsAsList.toList.foreach { item =>
          val divergence = contrastiveDivergence(batch.numExamples, item)

          nn = nn.updateWeights(
            calculateDiff(divergence._2, item, divergence._1, batch.numExamples, iteration)
          )
        }
    }

    nn
  }

  def contrastiveDivergence(inputLength:Int, input: DoubleMatrix) = {
    val gibbs = new GibbsSampler(nn)
    val numHidden = nn.w.rows
    val numVisible = nn.w.columns

    println(input.rows, input.columns)
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

    (g, inputSample)
  }

  def calculateDiff(inputSample: GibbsSample, input: DoubleMatrix, g: GibbsHVHSample, inputLength: Int, iteration: Int) = {
    val rate = learningRate(iteration) / inputLength
    val weights = inputSample.mean
      .mmul(input.transpose)
      .sub(g.hvMean.mmul(g.vhSample.transpose))
      .mul(rate)
      .transpose

    val hBias = inputSample.sample.sub(g.hvMean).mul(rate)
    val vBias = input.sub(g.vhSample).mul(rate)

    (weights, hBias, vBias)
  }
}







