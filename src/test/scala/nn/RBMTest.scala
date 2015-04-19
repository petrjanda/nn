package nn

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

    val rbm: RBM = new RBM(3, 2)

    ContrastiveDivergenceTrainer(
      nn = rbm,
      iterations = 1000,
      learningRate = 0.1,
      k = 2
    ).train(MatBuilder(3, 3, trainSet))

    rbm.reconstructM(testSetMat) should equal(
      new DoubleMatrix(3, 4,
        0.012769545530355501, 0.3116374332230691, 0.9922074530975166,
        0.01993776921249075, 0.32297561926066903, 0.9858651678750088,
        0.01993776921249075, 0.32297561926066903, 0.9858651678750088,
        0.01993776921249075, 0.32297561926066903, 0.9858651678750088
      )
    )

    import scala.collection.JavaConversions._

    rbm.reconstruct(testSet) should equal(rbm.reconstructM(testSetMat).columnsAsList.toList.map(_.toArray).toArray)

    rbm.propagateDown(Array(1, 0), 0) should equal(0.030049720410530306)
    rbm.propagateDown(Array(1, 0), 1) should equal(0.3327355092942503)
    rbm.propagateDown(Array(1, 0), 2) should equal(0.9750073350989872)

    rbm.propagateDownM(new DoubleMatrix(2, 1, 1, 0)) should equal(
      new DoubleMatrix(3, 1, 0.030049720410530306, 0.3327355092942503, 0.9750073350989872)
    )
  }
}