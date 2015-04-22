package nn

import nn.ds.DataSet
import nn.fn.obj.CrossEntropyError
import nn.trainers.ContrastiveDivergenceTrainer
import nn.utils.MatBuilder
import org.jblas.DoubleMatrix
import org.scalatest.{FlatSpec, FreeSpec, Matchers}

import scala.util.Random

class RBMTest extends FlatSpec with Matchers {
  it should "RBM" in {
    implicit val rng: Random = new Random(1234)

    case class TestSet(features:DoubleMatrix, targets:DoubleMatrix) extends DataSet { self =>
      def cp(features: DoubleMatrix, targets: DoubleMatrix): DataSet = self.copy(features, targets)
    }

    def buildSet(list:Array[Array[Double]]) = TestSet(MatBuilder(list.length, list.head.length, list), MatBuilder(2,1,Array(1.0, 2.0)))

    val trainSet = buildSet(Array(
      Array(0, 1, 1),
      Array(0, 1, 1),
      Array(0, 1, 1)
    ))

    val testSet = buildSet(Array(
      Array(0, 1, 1)
    ))

    val trainer = ContrastiveDivergenceTrainer(
      nn = RBM(3, 2, CrossEntropyError),
      iterations = 1000,
      evalIterations = 10000,
      miniBatchSize = 5,
      numParallel = 1,
      learningRate = 0.1,
      k = 2
    )

    val rbm = trainer.train(trainSet)

    println(rbm.loss(testSet))

    rbm.reconstruct(testSet) should equal(
      new DoubleMatrix(3, 1,
        0.013665678821389405, 0.9861983305985876, 0.9853362760435744
      )
    )
  }
}