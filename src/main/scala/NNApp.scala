import java.awt.image.Raster
import java.io.File
import javax.imageio.ImageWriter

import com.sksamuel.scrimage.{RGBColor, Image, Format}
import ds.salary.DemographicDataSet
import nn._
import nn.ds.DummyDataSet
import nn.fn.WeightDecay
import nn.fn.act.{HyperbolicTangent, Logistic}
import nn.fn.lrn.AnnealingRate
import nn.fn.obj.CrossEntropyError
import nn.fn.scr.{AbsoluteDiffScore, BinaryClassificationScore}
import nn.trainers.ContrastiveDivergenceTrainer
import nn.trainers.backprop.BackpropagationTrainer
import nn.utils.Repository
import org.jblas.DoubleMatrix
import nn.utils.Matrices._

import scala.util.Random

object NNApp extends App {
  val net :: op :: tail = args.toList

  net match {
    case "ff" => {
      op match {

        case "train" => {
          val trainingPath :: Nil = tail
          val trainingSet = DemographicDataSet(trainingPath)

          val nn = FeedForwardNN(
            Layer(trainingSet.numInputs, 10, HyperbolicTangent) :+
              Layer(trainingSet.numOutputs, Logistic),

            objective = CrossEntropyError,
            score = BinaryClassificationScore(.6),
            weightDecay = WeightDecay(0.0)
          )

          BackpropagationTrainer(
            nn = nn,
            numIterations = 50000,
            miniBatchSize = 500,
            learningRate = AnnealingRate(.1, 20000),
            evalIterations = 2000,
            momentumMultiplier = 0.2
          ).train(trainingSet)

          Repository.save(nn, "data/salary/net/ffn.o")
        }

        case "run" => {
          val testingPath :: Nil = tail
          val testingSet = DemographicDataSet(testingPath)

          val nn = Repository.load[FeedForwardNN]("data/salary/net/ffn.o")

          println(nn.eval(testingSet))
        }
      }
    }

    case "rbm" => {
      op match {
        case "train" => {
          implicit val rng = new Random(System.currentTimeMillis())

          println("--> Loading MNIST dataset")
          val mnist = MNIST.read("data/mnist/train-labels-idx1-ubyte", "data/mnist/train-images-idx3-ubyte", Some(100))
          val trainSet = DummyDataSet(mnist._1, mnist._2)
          println(s"--> ${trainSet.numExamples} samples loaded")

//          val trainSet = DemographicDataSet("data/salary/adult.data")

          println(s"--> Starting Contrastive Divergence trainer")
          val trainer:ContrastiveDivergenceTrainer = ContrastiveDivergenceTrainer(
            nn = RBM(trainSet.numInputs, 128, AbsoluteDiffScore, CrossEntropyError),
            iterations = 50000,
            evalIterations = 1000,
            miniBatchSize = 20,
            numParallel = 1,
            learningRate = AnnealingRate(.5, 50000),
            k = 1
          )
          println(s"--> trainer: ${trainer.toString}")
          println(s"--> network: ${trainer.nn.toString}")

          val nn = trainer.train(trainSet, { (i, nn) =>
            mat2img(nn.w.getColumns(Range(0, 20).toArray).data.toList, 1).write(new File(s"train-$i.png"))
          })

          println(s"--> Saving network to 'data/salary/net/rbm.o'")
          Repository.save(nn, "data/salary/net/rbm.o")

          println(s"--> Writing weights to the JSON file")
          writeToFile("weights.txt", printMat(nn.w))

          println(s"--> Done.")
        }

        case "run" => {

          val nn = Repository.load[RBM]("data/salary/net/rbm.o")
//          val testSet = DemographicDataSet("data/salary/adult.test")

          val mnist = MNIST.read("data/mnist/t10k-labels-idx1-ubyte", "data/mnist/t10k-images-idx3-ubyte", Some(20))
          val testSet = DummyDataSet(mnist._1, mnist._2)

          import scala.collection.JavaConversions._

          val in = testSet.features
          val out = nn.reconstruct(testSet)
          val pixels = in.columnsAsList().toList.zip(out.columnsAsList().toList).map {
            case (a, b) => a.data.toList.grouped(28).toList.zip(b.data.toList.grouped(28).toList).map {
              case (a, b) => a ::: b
            }.flatten
          }

          mat2img(pixels.flatten, 2).write(new File("in.png"))



//          println(nn.loss(testSet))
        }
      }
    }
  }

  def mat2img(m: List[Double], w:Int) = {
    val max = m.map(_.abs).max

    val data = m.map { i => {
      val c = 127 + (255 * i / max / 2).toInt

      RGBColor(c, c, c, 255).toInt
    } }

    Image(w*28, m.length / (w*28), data.toArray)
  }

  def printMat(mat:DoubleMatrix) = {
    s"{r:${mat.rows},c:${mat.columns},d:[${mat.data.toList.map(i => (i * 10000).round / 10000.0).mkString(",")}]"
  }

  def writeToFile(p: String, s: String): Unit = {
    val pw = new java.io.PrintWriter(new File(p))
    try pw.write(s) finally pw.close()
  }
}



