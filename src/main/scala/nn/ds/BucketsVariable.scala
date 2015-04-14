package nn.ds

case class BucketsVariable[T](options:T*) extends Variable[T] {
  val size: Int = options.size

  def apply(v:T):List[Double] = {
    val index = options.indexOf(v)

    if(index == -1) throw new Exception(v.toString)

    List.fill(index)(0.0) ++ List(1.0) ++ List.fill(size - index - 1)(0.0)
  }
}
