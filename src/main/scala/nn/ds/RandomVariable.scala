package nn.ds

trait RandomVariable[T] {
  def size:Int

  def apply(v:T):List[Double]
}
