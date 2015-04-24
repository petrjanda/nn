package nn.utils

import java.io.{FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream}

object Repository {
  def save[T](network: T, file: String) {
    val os = new ObjectOutputStream(new FileOutputStream(file))
    try {
      os.writeObject(network)
    } finally {
      os.close()
    }
  }

  def load[T](file: String): T = {
    val is = new ObjectInputStream(new FileInputStream(file))
    try {
      is.readObject().asInstanceOf[T]
    } finally {
      is.close()
    }
  }
}
