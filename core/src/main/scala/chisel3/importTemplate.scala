// See LICENSE for license details.

package chisel3

import chisel3.experimental.BaseModule

import scala.reflect.ClassTag

object importTemplate {

  def apply[T <: BaseModule](packge: Option[GeneratorPackageCreator])(implicit classTag: ClassTag[T]): Seq[Template[T]] = {
    internal.Builder.stash match {
      case Some(stash) => stash.importTemplates(classTag.runtimeClass.asInstanceOf[Class[T]], packge)
      case None => throw new ChiselException(s"No stash exists to lookup ${classTag.runtimeClass}!")
    }
  }


  def apply[T <: BaseModule](f: Seq[Template[T]] => Template[T], packge: Option[GeneratorPackageCreator] = None)(implicit classTag: ClassTag[T]): Template[T] = {
    f(apply(packge))
  }

  def apply[T <: BaseModule](packge: GeneratorPackageCreator)(implicit classTag: ClassTag[T]): Seq[Template[T]] = {
    apply(Some(packge))
  }

  def apply[T <: BaseModule]()(implicit classTag: ClassTag[T]): Seq[Template[T]] = {
    apply(None)
  }
}
