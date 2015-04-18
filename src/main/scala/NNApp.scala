import ds.salary.DemographicDataSet
import nn._
import nn.fn.WeightDecay
import nn.fn.act.{HyperbolicTangent, Logistic}
import nn.fn.lrn.AnnealingRate
import nn.fn.obj.CrossEntropyError
import nn.fn.scr.BinaryClassificationScore
import nn.trainers.backprop.Trainer
import org.jblas.MatrixFunctions._

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



}

import scala.util.Random




