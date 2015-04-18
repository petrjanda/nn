package nn.ds

case class ContinuousVariableWithLimits[T](comp:(T, T) => Boolean, limits:T*) extends Variable[T] {
  def size = limits.size + 1

  def apply(v:T):List[Double] = {
    val index = limits.zipWithIndex.collectFirst {
      case (limit, i) if comp(limit, v) => i
    }.getOrElse(size - 1)

    List.fill(index)(0.0) ++ List(1.0) ++ List.fill(size - index - 1)(0.0)
  }
}

case class ContinuousVariable(min:Int, max:Int) {
  def size = 1

  def range = max - min

  def apply(v:Int):List[Double] = List.fill(1)(((v - min) / range).toDouble)
}