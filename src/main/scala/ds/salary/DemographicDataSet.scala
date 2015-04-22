package ds.salary

import org.jblas.DoubleMatrix
import nn.ds.DataSet

import scala.util.{Try, Success}

class DemographicDataSet(val features:DoubleMatrix, val targets:DoubleMatrix) extends DataSet {
  def cp(inputs: DoubleMatrix, targets: DoubleMatrix): DataSet =
    new DemographicDataSet(inputs, targets)
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
      new DoubleMatrix(Sample.inputSize, length, samples.map(_.toInput).flatten:_*)

    def targets: DoubleMatrix = new DoubleMatrix(1, length, samples.map(_.toTarget).flatten:_*)

    new DemographicDataSet(inputs, targets)
  }
}