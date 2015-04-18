package nn

import nn.trainers.ContrastiveDivergenceTrainer
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

    val rbm: RBM = new RBM(3, 1)

    ContrastiveDivergenceTrainer(
      nn = rbm,
      iterations = 1000,
      learningRate = 0.1,
      k = 2
    ).train(trainSet)

    rbm.reconstruct(testSet) should equal(
      Array(
        Array(0.006165282894259764, 0.006397587880907305, 0.9947420420073434),
        Array(0.0073350521633475455, 0.007619206686898068, 0.9934712649173539),
        Array(0.007352346565525056, 0.0076372762357421594, 0.9934520802194916)
      )
    )
  }

}