package chisel3
import scala.language.dynamics


trait Interface extends Dynamic {
  def selectDynamic(name: String) = {
    ???
  }
}

object Interface {
  def apply[A <: Interface](i: A): A = i
}