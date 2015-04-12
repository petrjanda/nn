import nn._
import nn.ds.DataSet
import nn.fn.WeightDecay
import nn.fn.act.{HyperbolicTangent, Logistic}
import nn.fn.lrn.ConstantRate
import nn.fn.obj.CrossEntropyError
import org.jblas.DoubleMatrix
import org.jblas.MatrixFunctions._

import scala.util.{Success, Try}

object NNApp extends App {

  val train = DemographicDataSet("data/salary/train.data")
  val test = DemographicDataSet("data/salary/test.data")

  val nn = NeuralNetwork(
    Layer(train.numInputs, 15, Logistic) :+
    Layer(8, Logistic) :+
    Layer(train.numOutputs, Logistic),

    objective = CrossEntropyError,
    weightDecay = WeightDecay(0.0)
  )

  println("diff: " + test.targets.sub(floor(nn.compute(test.inputs).add(.5))).sum)

  Trainer(
    numIterations = 15000,
    miniBatchSize = 1000,
    learningRate = ConstantRate(.5),
    evalIterations = 500,
    momentumMultiplier = 0.1
  ).train(nn, train)

//  println("results: " + floor(nn.compute(test.inputs).add(.5)))
//  println("targets:" + test.targets)
  println("diff: " + test.targets.sub(floor(nn.compute(test.inputs).add(.5))).sum)
}

trait Var[T] {
  def size:Int

  def apply(v:T):List[Double]
}

case class ContinuousVariable[T](comp:(T, T) => Boolean, limits:T*) extends Var[T] {
  def size = limits.size + 1

  def apply(v:T):List[Double] = {
    val index = limits.zipWithIndex.collectFirst {
      case (limit, i) if comp(limit, v) => i
    }.getOrElse(size - 1)

    List.fill(index)(0.0) ++ List(1.0) ++ List.fill(size - index - 1)(0.0)
  }
}

case class Variable[T](options:T*) extends Var[T] {
  val size: Int = options.size

  def apply(v:T):List[Double] = {
    val index = options.indexOf(v)

    if(index == -1) throw new Exception(v.toString)

    List.fill(index)(0.0) ++ List(1.0) ++ List.fill(size - index - 1)(0.0)
  }
}

object Sample {
  val Age = ContinuousVariable((a:Int, b:Int) => a > b, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
  val Workclass = Variable("?", "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked")
  val Education = Variable("?", "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool")
  val Occupation = Variable("?", "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces")
  val NativeCountry = Variable("?", "United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands")
  val CapitalGain = ContinuousVariable((a:Int, b:Int) => a > b, 0, 1000, 5000, 10000, 50000)
  val Sex = Variable("?", "Male", "Female")
  val HoursPerWeek = ContinuousVariable((a:Int, b:Int) => a > b, 0, 10, 20, 30, 40, 50, 60)
  val Race = Variable("White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black")

  val size = Sample.Age.size +
    Sample.Workclass.size +
    Sample.Occupation.size +
    Sample.Education.size +
    Sample.NativeCountry.size +
    Sample.CapitalGain.size +
    Sample.Sex.size +
    Sample.HoursPerWeek.size +
    Sample.Race.size
}

case class Sample(age:String, workclass:String, flnwgt:String, education:String, educationNum:String,
                  maritalStatus:String, occupation:String, relationship:String, race:String, sex:String,
                  capitalGain:String, capitalLoss:String, hoursPerWeek:String, nativeCountry:String, target:String) {
  def toInput: List[Double] =
    Sample.Age(age.toInt) ++
    Sample.Workclass(workclass) ++
    Sample.Occupation(occupation) ++
    Sample.Education(education) ++
    Sample.NativeCountry(nativeCountry) ++
    Sample.CapitalGain(capitalGain.toInt) ++
    Sample.Sex(sex) ++
    Sample.HoursPerWeek(hoursPerWeek.toInt) ++
    Sample.Race(race)

  def toTarget: List[Double] = {
    List(if(target == ">50K") 1.0 else 0.0)
  }
}

object DemographicDataSet {
  def apply(path:String) = {
    lazy val lines = scala.io.Source.fromFile(path).getLines().toList

    lazy val samples = lines.map { line =>
      Try {
        line.split(", ").toList match {
          case List(age, workclass, flnwgt, education, educationNum,
            maritalStatus, occupation, relationship, race, sex,
            capitalGain, capitalLoss, hoursPerWeek, nativeCountry, target
          ) =>
            Sample(
              age, workclass, flnwgt, education, educationNum,
              maritalStatus, occupation, relationship, race, sex,
              capitalGain, capitalLoss, hoursPerWeek, nativeCountry, target
            )
        }
      }
    }.collect { case Success(k) => k }

    lazy val length = samples.length

    def inputs: DoubleMatrix =
      new DoubleMatrix(Sample.size, length, samples.map(_.toInput).flatten:_*)

    def targets: DoubleMatrix = new DoubleMatrix(1, length, samples.map(_.toTarget).flatten:_*)

    new DemographicDataSet(inputs, targets)
  }
}

class DemographicDataSet(val inputs:DoubleMatrix, val targets:DoubleMatrix) extends DataSet {
  def copy(inputs: DoubleMatrix, targets: DoubleMatrix): DataSet =
    new DemographicDataSet(inputs, targets)
}



// fnlwgt: continuous.
// marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
// relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
// race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
// capital-loss: continuous.