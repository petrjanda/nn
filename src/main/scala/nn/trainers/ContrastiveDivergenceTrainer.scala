package nn.trainers

import nn.ds.DataSet
import nn.trainers.gibbs.{GibbsHVHSample, GibbsSample, GibbsSampler}
import nn.{NeuralNetwork, RBM}
import org.jblas.DoubleMatrix

import scala.util.Random

case class ContrastiveDivergenceTrainer(private var nn:RBM, iterations:Int, evalIterations:Int, miniBatchSize:Int, numParallel:Int, learningRate:Double, k:Int)(implicit rng:Random) {
  import scala.collection.JavaConversions._

  def evalIteration(iteration: Int, trainingSet: DataSet) {
    if ((iteration - 1) % evalIterations == 0) {
      val loss = nn.loss(trainingSet)
      println("Iteration:%5d, Loss: %.5f".format(iteration + 1, loss))
    }
  }

  def train(dataSet:DataSet):RBM = {
    dataSet.miniBatches(miniBatchSize).grouped(numParallel).take(iterations).zipWithIndex.foreach {
      case (batches, iteration) =>
        val batch = batches(0)

        evalIteration(iteration, dataSet)

        batch.inputs.columnsAsList.toList.foreach { item =>
          nn = nn.updateWeights(
            contrastiveDivergence(batch.numExamples, item)
          )
        }
    }

    nn
  }

  def train(inputs:DoubleMatrix):RBM = {
    0.until(iterations).foreach { _ =>
      inputs.columnsAsList.toList.foreach { item =>
        nn = nn.updateWeights(
          contrastiveDivergence(inputs.columns, item)
        )
      }
    }

    nn
  }

  def contrastiveDivergence(inputLength:Int, input: DoubleMatrix) = {
    val gibbs = new GibbsSampler(nn)
    val numHidden = nn.w.rows
    val numVisible = nn.w.columns

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







