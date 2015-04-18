package nn

import nn.trainers.ContrastiveDivergenceTrainer
import org.jblas.DoubleMatrix
import org.scalatest.{FreeSpec, Matchers}

import scala.util.Random

class RBMTest extends FreeSpec with Matchers {
  "RBM" - {
    implicit val rng: Random = new Random(1234)

    val trainSet: Array[Array[Int]] = Array(
      Array(0, 0, 1),
      Array(0, 0, 1)
    )

    val testSet = Array(
      Array(0, 0, 1),
      Array(1, 0, 1),
      Array(0, 1, 1)
    )

    val testSetMat = MatBuilder(3, 3, testSet)

    val rbm: RBM = new RBM(3, 1)

    ContrastiveDivergenceTrainer(
      nn = rbm,
      iterations = 1000,
      learningRate = 0.1,
      k = 2
    ).train(trainSet)

    val s = System.currentTimeMillis()
    Range(0, 100000).foreach { i =>
      rbm.reconstruct(testSetMat)
    }
    println(System.currentTimeMillis() - s)

    val s2 = System.currentTimeMillis()
    Range(0, 100000).foreach { i =>
      rbm.reconstructOld(testSet)
    }
    println(System.currentTimeMillis() - s2)
  }
}