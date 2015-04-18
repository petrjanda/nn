package nn.trainers.gibbs

case class GibbsHVHSample(vhMean:Array[Double], vhSample:Array[Int], hvMean:Array[Double], hvSample:Array[Int])

object GibbsHVHSample {
  def apply(vh:GibbsSample, hv:GibbsSample): GibbsHVHSample = {
    GibbsHVHSample(vh.mean, vh.sample, hv.mean, hv.sample)
  }
}