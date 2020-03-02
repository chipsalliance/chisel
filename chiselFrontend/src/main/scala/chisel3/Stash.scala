// See LICENSE for license details.

package chisel3.incremental

import java.io._

import _root_.firrtl.annotations.NoTargetAnnotation
import _root_.firrtl.options.{HasShellOptions, ShellOption, Unserializable}
import chisel3.{GeneratorPackage, GeneratorPackageCreator, RawModule, Template}
import chisel3.experimental.BaseModule

import scala.collection.mutable

/** A stash contains all caches available to a running Chisel elaboration
  * Every elaboration can only have one Stash
  *
  * After elaboration, a Stash can export its state as a new Cache
  *
  */
case class Stash(stash: Map[GeneratorPackageCreator, GeneratorPackage[BaseModule]]) extends NoTargetAnnotation with Unserializable {

  val builderModules: mutable.HashMap[Class[_], Seq[Template[_]]] = mutable.HashMap.empty

  def store[T <: RawModule](module: T): Unit = {
    builderModules(module.getClass) = Template(module, None) +: builderModules.getOrElse(module.getClass, Seq.empty[Template[T]])
  }

  def importTemplates[T <: BaseModule](clazz: Class[T], packageName: Option[GeneratorPackageCreator]): Seq[Template[T]] = {
    if(packageName.nonEmpty) {
      val packge = stash(packageName.get)
      packge.find(clazz)
    } else {
      builderModules.getOrElse(clazz, Seq.empty[Template[T]]).map(_.asInstanceOf[Template[T]])
    }
  }
}
