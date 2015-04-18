package nn

import scala.util.Random

class RBM(val N: Int, val n_visible: Int, val n_hidden: Int, var rng: Random) {

  val a: Double = 1 / n_visible
  var W: Array[Array[Double]] = Array.ofDim[Double](n_hidden, n_visible)
  var hbias: Array[Double] = Array.fill(n_hidden) { 0.0 }
  var vbias: Array[Double] = Array.fill(n_visible) { 0.0 }

  Range(0, n_hidden).foreach { i => 
    Range(0, n_visible).foreach { j =>
      W(i)(j) = Fn.uniform(-a, a, rng)
    }
  }

  def propagateUp(v: Array[Int], w: Array[Double], b: Double): Double = {
    Fn.sigmoid(
      Range(0, n_visible).toArray.foldLeft(0.0) { (t, j) => t + w(j) * v(j) } + b
    )
  }

  def propagateDown(h: Array[Int], i: Int, b: Double): Double = {
    Fn.sigmoid(
      Range(0, n_hidden).toArray.foldLeft(0.0) { (t, j) => t + W(j)(i) * h(j) } + b
    )
  }

  def reconstruct(v: Array[Array[Int]]): Array[Array[Double]] = {
    v.map { v =>
      val h = Range(0, n_hidden).toArray.map { i =>
        propagateUp(v, W(i), hbias(i))
      }

      val layer = Layer(n_hidden, vbias, W, h)

      Range(0, n_visible).toArray.map { layer.activationOutput(_) }
    }
  }

  case class Layer(n_hidden: Int, vbias:Array[Double], W: Array[Array[Double]], h: Array[Double]) {
    def activationOutput(i: Int) = {
      Fn.sigmoid(
        0.until(n_hidden).foldLeft(0.0) { (t, j) => t + W(j)(i) * h(j) } + vbias(i)
      )
    }
  }
}



object Fn {
  def uniform(min: Double, max: Double, rng:Random): Double = rng.nextDouble() * (max - min) + min
  def binomial(n: Int, p: Double, rng:Random): Int = {
    if(p < 0 || p > 1) return 0

    var c: Int = 0
    var r: Double = 0

    var i: Int = 0
    for(i <- 0 until n) {
      r = rng.nextDouble()
      if(r < p) c += 1
    }

    c
  }

  def sigmoid(x: Double): Double = 1.0 / (1.0 + math.pow(math.E, -x))
}


case class ContrastiveDivergenceTrainer(iterations:Int, learningRate:Double, k:Int) {
  def train(rbm:RBM, dataSet:Array[Array[Int]]) = {
    0.until(iterations).foreach { _ =>
      dataSet.foreach { item =>
        contrastive_divergence(rbm, item, learningRate, k)
      }
    }
  }

  def contrastive_divergence(rbm:RBM, input: Array[Int], lr: Double, k: Int) {
    val nv_means: Array[Double] = new Array[Double](rbm.n_visible)
    val nv_samples: Array[Int] = new Array[Int](rbm.n_visible)
    var nh_means: Array[Double] = new Array[Double](rbm.n_hidden)
    var nh_samples: Array[Int] = new Array[Int](rbm.n_hidden)

    /* CD-k */
    val (ph_mean, ph_sample) = sampleHGivenV(rbm, input)

    0.until(k).foreach { step =>
      val sample = if(step == 0) ph_sample else nh_samples

      val gibbs = gibbs_hvh(rbm, sample, nv_means, nv_samples)

      nh_means = gibbs._1
      nh_samples = gibbs._2
    }

    0.until(rbm.n_hidden).foreach { i =>
      0.until(rbm.n_visible).foreach { j =>
        rbm.W(i)(j) += lr * (ph_mean(i) * input(j) - nh_means(i) * nv_samples(j)) / rbm.N
      }

      rbm.hbias(i) += lr * (ph_sample(i) - nh_means(i)) / rbm.N
    }


    0.until(rbm.n_visible).foreach { i =>
      rbm.vbias(i) += lr * (input(i) - nv_samples(i)) / rbm.N
    }
  }

  def gibbs_hvh(rbm:RBM, h0_sample: Array[Int], nv_means: Array[Double], nv_samples: Array[Int]) = {
    sampleVGivenH(rbm:RBM, h0_sample, nv_means, nv_samples)
    sampleHGivenV(rbm:RBM, nv_samples)
  }

  def sampleHGivenV(rbm:RBM, v0_sample: Array[Int]) = {
    val mean = 0.until(rbm.n_hidden).map { i => rbm.propagateUp(v0_sample, rbm.W(i), rbm.hbias(i)) }.toArray
    val sample = mean.map { s => Fn.binomial(1, s, rbm.rng) }

    (mean, sample)
  }

  def sampleVGivenH(rbm:RBM, h0_sample: Array[Int], mean: Array[Double], sample: Array[Int]) {
    var i: Int = 0
    for(i <- 0 until rbm.n_visible) {
      mean(i) = rbm.propagateDown(h0_sample, i, rbm.vbias(i))
      sample(i) = Fn.binomial(1, mean(i), rbm.rng)
    }
  }
}

