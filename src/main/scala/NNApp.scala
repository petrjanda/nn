import java.io.File

import ds.salary.DemographicDataSet
import nn._
import nn.fn.WeightDecay
import nn.fn.act.{HyperbolicTangent, Logistic}
import nn.fn.lrn.AnnealingRate
import nn.fn.obj.CrossEntropyError
import nn.fn.scr.BinaryClassificationScore
import nn.trainers.ContrastiveDivergenceTrainer
import nn.trainers.backprop.Trainer
import nn.utils.Repository
import org.jblas.DoubleMatrix
import org.jblas.MatrixFunctions._

import scala.util.Random

object NNApp extends App {

//  val train = DemographicDataSet("data/salary/adult.data")
//  val test = DemographicDataSet("data/salary/adult.test")
//
//  val nn = NeuralNetwork(
//    Layer(train.numInputs, 10, HyperbolicTangent) :+
//    Layer(train.numOutputs, Logistic),
//
//    objective = CrossEntropyError,
//    score = BinaryClassificationScore(.6),
//    weightDecay = WeightDecay(0.0)
//  )
//
//  val base = nn.eval(test)
//  println(s"Iteration: base, Accuracy: $base")
//
//  Trainer(
//    numIterations = 20000,
//    miniBatchSize = 1000,
//    learningRate = AnnealingRate(.1, 20000),
//    evalIterations = 2000,
//    momentumMultiplier = 0.2
//  ).train(nn, train)
//
//  val testing = nn.eval(test)
//
//  println(s"Iteration: test, Accuracy: $testing")
//
//  println(nn.layers.head.weights.getRow(0))
//  println(nn.layers.tail.head.weights)


  implicit val rng = new Random(System.currentTimeMillis())

  val trainSet = DemographicDataSet("data/salary/adult.data")

  val nn = new RBM(trainSet.numInputs, 50)
  ContrastiveDivergenceTrainer(nn, 1000, 0.15, 2).train(trainSet)

  Repository.save(nn, "data/salary/net/rbm.o")


  val nn2 = Repository.load[RBM]("data/salary/net/rbm.o")
  val testSet = DemographicDataSet("data/salary/adult.test")

  writeToFile("weights.txt", printMat(nn2.wmat))

//  println(BinaryClassificationScore(.5).score(testSet.inputs, nn.reconstructM(testSet)))


  def printMat(mat:DoubleMatrix) = {
    mat.data.toList.map(i => (i * 10000).round / 10000.0).toString
  }

  def writeToFile(p: String, s: String): Unit = {
    val pw = new java.io.PrintWriter(new File(p))
    try pw.write(s) finally pw.close()
  }
}


import scala.util.Random




