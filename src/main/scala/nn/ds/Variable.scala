package nn.ds

trait Variable[T] {
  def size:Int

  def apply(v:T):List[Double]
}
