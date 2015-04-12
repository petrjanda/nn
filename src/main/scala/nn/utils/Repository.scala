package nn.utils

import java.io.{FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream}

import nn.NeuralNetwork

object Repository {
  def save(network: NeuralNetwork, file: String) {
    val os = new ObjectOutputStream(new FileOutputStream(file))
    try {
      os.writeObject(network)
    } finally {
      os.close()
    }
  }

  def load(file: String): NeuralNetwork = {
    val is = new ObjectInputStream(new FileInputStream(file))
    try {
      is.readObject().asInstanceOf[NeuralNetwork]
    } finally {
      is.close()
    }
  }
}
