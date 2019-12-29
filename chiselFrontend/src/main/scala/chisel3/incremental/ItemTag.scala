package chisel3.incremental

import java.io.File

import chisel3.experimental.BaseModule

/** An elaboration-agnostic tag for an elaborated module
  *
  * Used to import previously elaborated modules into a new elaboration
  *
  * @param chiselClass
  * @param parameters
  * @tparam T
  */
case class ItemTag[+T <: BaseModule](chiselClass: String, parameters: Seq[AnyRef]) {
  def tag: String = parameters.hashCode().toHexString

  def tagFileName: String = chiselClass + "." + tag + ".tag"

  def itemFileName: String = chiselClass + "." + tag + ".item"

  private[chisel3] def store(directory: String, module: BaseModule): Unit = {
    Stash.store(new File(directory + itemFileName), module)
    Stash.store(new File(directory + tagFileName), this)
  }

  private[chisel3] def load(directory: String): Option[T] = {
    Stash.load[T](new File(directory + itemFileName))
  }
}
