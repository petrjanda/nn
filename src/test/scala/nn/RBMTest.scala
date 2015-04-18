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
      Array(0, 0, 1),
      Array(0, 1, 1)
    )

    val testSet = Array(
      Array(0, 0, 1),
      Array(1, 0, 1),
      Array(0, 1, 1)
    )

    val testSetMat = MatBuilder(3, 3, testSet)

    val rbm: RBM = new RBM(3, 2)

    ContrastiveDivergenceTrainer(
      nn = rbm,
      iterations = 1000,
      learningRate = 0.1,
      k = 2
    ).train(trainSet)

    rbm.reconstructM(testSetMat) should equal(
      Array(
        Array(0.003329569358668085, 0.2951388824444748, 0.9978182978208314),
        Array(0.007276932788137207, 0.31074279541024996, 0.9939388505529718),
        Array(0.0034921383069850156, 0.2960706903484258, 0.9976774927884484)
      )
    )

    rbm.reconstructM(testSetMat) should equal(rbm.reconstructM(testSetMat))

    rbm.propagateDown(Array(1, 0), 0) should equal(.010427013567532374)
    rbm.propagateDown(Array(1, 0), 1) should equal(.3172730557439495)
    rbm.propagateDown(Array(1, 0), 2) should equal(.9899104140658883)

    rbm.propagateDownM(new DoubleMatrix(2, 1, 1, 0)) should equal(
      new DoubleMatrix(3, 1, 0.010427013567532372, 0.3172730557439495, 0.9899104140658883)
    )
  }
}