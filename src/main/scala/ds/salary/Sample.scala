package ds.salary

import nn.ds.{ContinuousVariable, ContinuousVariableWithLimits, BucketsVariable}

object Sample {
//  val Age = ContinuousVariable(0, 100)
  val Age = ContinuousVariable(0, 100)
  val Workclass = BucketsVariable("?", "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked")
  val Education = BucketsVariable("?", "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool")
  val Occupation = BucketsVariable("?", "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces")
  val NativeCountry = BucketsVariable("?", "United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands")
  val CapitalGain = ContinuousVariableWithLimits((a:Int, b:Int) => a > b, 0, 1000, 5000, 10000, 50000)
  val Sex = BucketsVariable("?", "Male", "Female")
  val HoursPerWeek = ContinuousVariableWithLimits((a:Int, b:Int) => a > b, 0, 10, 20, 30, 40, 50, 60)
  val Race = BucketsVariable("White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black")

  val inputSize = Sample.Age.size +
//    Sample.Workclass.size +
//    Sample.Occupation.size +
//    Sample.Education.size +
//    Sample.NativeCountry.size +
//    Sample.CapitalGain.size +
//    Sample.Sex.size +
//    Sample.HoursPerWeek.size +
    Sample.Race.size
}

case class Sample(age:String, workclass:String, flnwgt:String, education:String, educationNum:String,
                  maritalStatus:String, occupation:String, relationship:String, race:String, sex:String,
                  capitalGain:String, capitalLoss:String, hoursPerWeek:String, nativeCountry:String, target:String) {
  def toInput: List[Double] =
    Sample.Age(age.toInt) ++
//      Sample.Workclass(workclass) ++
//      Sample.Occupation(occupation) ++
//      Sample.Education(education) ++
//      Sample.NativeCountry(nativeCountry) ++
//      Sample.CapitalGain(capitalGain.toInt) ++
//      Sample.Sex(sex) ++
//      Sample.HoursPerWeek(hoursPerWeek.toInt) ++
      Sample.Race(race)

  def toTarget: List[Double] = {
    List(if(target == ">50K" || target == ">50K.") 1.0 else 0.0)
  }
}

