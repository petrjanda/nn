import java.io.File

import ds.salary.DemographicDataSet
import nn._
import nn.fn.WeightDecay
import nn.fn.act.{HyperbolicTangent, Logistic}
import nn.fn.lrn.AnnealingRate
import nn.fn.obj.CrossEntropyError
import nn.fn.scr.BinaryClassificationScore
import nn.trainers.ContrastiveDivergenceTrainer
import nn.trainers.backprop.BackpropagationTrainer
import nn.utils.Repository
import org.jblas.DoubleMatrix

import scala.util.Random

object NNApp extends App {
  val net :: op :: Nil = args.toList

  net match {
    case "ff" => {
        val trainingSet = DemographicDataSet("data/salary/adult.data")
        val test = DemographicDataSet("data/salary/adult.test")

        val nn = FeedForwardNN(
          Layer(trainingSet.numInputs, 10, HyperbolicTangent) :+
          Layer(trainingSet.numOutputs, Logistic),

          objective = CrossEntropyError,
          score = BinaryClassificationScore(.6),
          weightDecay = WeightDecay(0.0)
        )

        val base = nn.eval(test)
        println(s"Iteration: base, Accuracy: $base")

        BackpropagationTrainer(
          nn = nn,
          numIterations = 50000,
          miniBatchSize = 500,
          learningRate = AnnealingRate(.1, 20000),
          evalIterations = 2000,
          momentumMultiplier = 0.2
        ).train(trainingSet)

        val testing = nn.eval(test)

        println(s"Iteration: test, Accuracy: $testing")

        println(nn.layers.head.weights.getRow(0))
        println(nn.layers.tail.head.weights)
    }

    case "rbm" => {
      op match {
        case "train" => {
          implicit val rng = new Random(System.currentTimeMillis())

          val trainSet = DemographicDataSet("data/salary/adult.data")

          val nn = ContrastiveDivergenceTrainer(
            nn = RBM(trainSet.numInputs, 100, BinaryClassificationScore(.5), CrossEntropyError),
            iterations = 50000,
            evalIterations = 100,
            miniBatchSize = 100,
            numParallel = 1,
            learningRate = 0.8,
            k = 2
          ).train(trainSet)

          Repository.save(nn, "data/salary/net/rbm.o")

          writeToFile("weights.txt", printMat(nn.w))
        }

        case "run" => {
          val nn2 = Repository.load[RBM]("data/salary/net/rbm.o")
          val testSet = DemographicDataSet("data/salary/adult.test")
//          val s = System.currentTimeMillis()
          val reconstructed = nn2.reconstruct(testSet)

          println(nn2.loss(testSet))
//          println(System.currentTimeMillis() - s)

//          println(BinaryClassificationScore(.5).score(testSet.inputs, reconstructed))
        }
      }
    }
  }

  def printMat(mat:DoubleMatrix) = {
    s"{r:${mat.rows},c:${mat.columns},d:[${mat.data.toList.map(i => (i * 10000).round / 10000.0).mkString(",")}]"
  }

  def writeToFile(p: String, s: String): Unit = {
    val pw = new java.io.PrintWriter(new File(p))
    try pw.write(s) finally pw.close()
  }
}



