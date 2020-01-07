package chisel3.incremental

import java.io.File

import chisel3.experimental.BaseModule

import scala.reflect.ClassTag
import scala.reflect.runtime.universe.TypeTag

trait Tag {
  val chiselClassName: String
  val parameters: Seq[Any]

  def tag: String = parameters.hashCode().toHexString

  def tagFileName: String = chiselClassName + "." + tag + ".tag"

  def itemFileName: String = chiselClassName + "." + tag + ".item"

  private[chisel3] def store(directory: String, module: BaseModule): Unit = {
    Stash.store(directory, itemFileName, module)
    Stash.store(directory, tagFileName, this)
  }

  private[chisel3] def untypedLoad(directory: String): Option[BaseModule] = {
    Stash.load[BaseModule](directory, itemFileName)
  }
}

case class UntypedTag(chiselClassName: String, parameters: Seq[Any]) extends Tag

/** An elaboration-agnostic tag for an elaborated module
  *
  * Used to import previously elaborated modules into a new elaboration
  *
  * @param parameters
  * @tparam T
  */
case class ItemTag[T <: BaseModule](parameters: Seq[Any])(implicit classTag: ClassTag[T]) extends Tag {
  val chiselClass = classTag.runtimeClass.asInstanceOf[Class[T]]
  val chiselClassName = chiselClass.getName

  private[chisel3] def load(directory: String): Option[T] = {
    Stash.load(directory, itemFileName).asInstanceOf[Option[T]]
  }
}
