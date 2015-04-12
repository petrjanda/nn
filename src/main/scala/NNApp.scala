import nn._
import nn.ds.{DataSet, DummyDataSet}
import nn.fn.{WeightDecay, LearningFunction}
import nn.fn.act.Logistic
import nn.fn.lrn.ConstantRate
import nn.fn.obj.CrossEntropyError
import org.jblas.DoubleMatrix

object NNApp extends App{
  val inputs = new DoubleMatrix(2, 2, 1, 1, 1, 1)
  val targets = new DoubleMatrix(2, 2, 1, 1, 1, 1)
  val ds = DummyDataSet(inputs, targets)

  val nn = NeuralNetwork(
    Layer(2, 2, Logistic) :+ Layer(2, Logistic),
    objective = CrossEntropyError,
    weightDecay = WeightDecay(0.0)
  )

  println(nn.errorGradients(ds))
}




class DemographicDataSet extends DataSet {

}

// age: continuous.
// workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
// fnlwgt: continuous.
// education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
// education-num: continuous.
// marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
// occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
// relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
// race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
// sex: Female, Male.
// capital-gain: continuous.
// capital-loss: continuous.
// hours-per-week: continuous.
// native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
// class: >50K, <=50K