package nn.trainers

import nn.ds.DataSet
import nn.fn.LearningFunction
import nn.trainers.gibbs.{GibbsHVHSample, GibbsSample, GibbsSampler}
import nn.{FeedForwardNN$, RBM}
import org.jblas.DoubleMatrix

import scala.util.Random

case class ContrastiveDivergenceTrainer(var nn:RBM, iterations:Int, evalIterations:Int, miniBatchSize:Int, numParallel:Int, learningRate:LearningFunction, k:Int)(implicit rng:Random) {
  def evalIteration(iteration: Int, trainingSet: DataSet, f:(Int, RBM) => Unit) {
    if ((iteration + 1) % evalIterations == 0) {
      f(iteration, nn)

      val loss = nn.loss(trainingSet)
      val score = nn.eval(trainingSet)
      println("Iteration:%5d, Loss: %.10f, Diff: %.10f".format(iteration + 1, loss, score))
    }
  }

  def train(dataSet:DataSet, f:(Int, RBM) => Unit):RBM = {
    evalIteration(-1, dataSet, f)

    dataSet.miniBatches(miniBatchSize).grouped(numParallel).take(iterations).zipWithIndex.foreach {
      case (batches, iteration) =>
        val batch = batches(0)

        evalIteration(iteration, dataSet, f)

        val divergence = contrastiveDivergence(batch.numExamples, batch.features)

        nn = nn.updateWeights(
          calculateDiff(divergence._2, batch.features, divergence._1, batch.numExamples, iteration)
        )
    }

    nn
  }

  def contrastiveDivergence(inputLength:Int, input: DoubleMatrix) = {
    val gibbs = new GibbsSampler(nn)
    val numHidden = nn.w.rows
    val numVisible = nn.w.columns

    val inputSample = gibbs.sampleHGivenV(input)

    val first = GibbsHVHSample(
      new DoubleMatrix(input.rows, numVisible).fill(0.0),
      new DoubleMatrix(input.rows, numHidden).fill(0.0),
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

    (weights, hBias.rowMeans, vBias.rowMeans)
  }

  override def toString = {
    s"iterations: $iterations, miniBatchSize: $miniBatchSize, numParallel: $numParallel, learningRate: $learningRate, k: $k"
  }
}







