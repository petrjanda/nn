import _root_.ds.salary.DemographicDataSet
import nn._
import nn.fn.WeightDecay
import nn.fn.act.{HyperbolicTangent, Logistic}
import nn.fn.lrn.AnnealingRate
import nn.fn.obj.CrossEntropyError
import org.jblas.MatrixFunctions._

object NNApp extends App {

  val train = DemographicDataSet("data/salary/adult.data")
  val test = DemographicDataSet("data/salary/adult.test")

  val nn = NeuralNetwork(
    Layer(train.numInputs, 5, HyperbolicTangent) :+
    Layer(train.numOutputs, Logistic),

    objective = CrossEntropyError,
    weightDecay = WeightDecay(0.0)
  )

  val base = test.targets.sub(floor(nn.compute(test.inputs).add(.5))).sum
  println(s"Iteration: base, Accuracy: $base")

  Trainer(
    numIterations = 1000,
    miniBatchSize = 1000,
    learningRate = AnnealingRate(.1, 5000),
    evalIterations = 1000,
    momentumMultiplier = 0.0
  ).train(nn, train)

  val testing = test.targets.sub(floor(nn.compute(test.inputs).add(.5))).sum

  println(s"Iteration: test, Accuracy: $testing")

  println(nn.layers.head.weights.getRow(0))
  println(nn.layers.tail.head.weights)
}

// fnlwgt: continuous.
// marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
// relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
// race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
// capital-loss: continuous.