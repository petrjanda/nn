package nn

import nn.fn.obj.CrossEntropyError
import nn.trainers.ContrastiveDivergenceTrainer
import nn.utils.MatBuilder
import org.jblas.DoubleMatrix
import org.scalatest.{FreeSpec, Matchers}

import scala.util.Random

class RBMTest extends FreeSpec with Matchers {
  "RBM" - {
    implicit val rng: Random = new Random(1234)

    val trainSet: Array[Array[Double]] = Array(
      Array(0, 0, 1),
      Array(0, 0, 1),
      Array(0, 1, 1)
    )

    val testSet: Array[Array[Double]] = Array(
      Array(0, 0, 1),
      Array(1, 0, 1),
      Array(1, 0, 1),
      Array(1, 0, 1)
    )

    val testSetMat = MatBuilder(4, 3, testSet)

    val trainer = ContrastiveDivergenceTrainer(
      nn = RBM(3, 2, CrossEntropyError),
      iterations = 1000,
      evalIterations = 100,
      miniBatchSize = 1000,
      numParallel = 1,
      learningRate = 0.1,
      k = 2
    )

    val rbm = trainer.train(MatBuilder(3, 3, trainSet))

    rbm.reconstruct(testSetMat) should equal(
      new DoubleMatrix(3, 4,
        0.0033295693586680845, 0.2951388824444747, 0.9978182978208314,
        0.0072769327881372055, 0.3107427954102499, 0.9939388505529718,
        0.0072769327881372055, 0.3107427954102499, 0.9939388505529718,
        0.0072769327881372055, 0.3107427954102499, 0.9939388505529718
      )
    )
  }
}