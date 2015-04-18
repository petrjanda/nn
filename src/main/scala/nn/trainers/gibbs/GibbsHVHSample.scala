package nn.trainers.gibbs

case class GibbsHVHSample(vhMean:Array[Double], vhSample:Array[Double], hvMean:Array[Double], hvSample:Array[Double])

object GibbsHVHSample {
  def apply(vh:GibbsSample, hv:GibbsSample): GibbsHVHSample = {
    GibbsHVHSample(vh.mean, vh.sample, hv.mean, hv.sample)
  }
}