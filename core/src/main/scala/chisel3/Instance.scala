// SPDX-License-Identifier: Apache-2.0

package chisel3

import scala.collection.immutable.ListMap
import scala.collection.mutable.{ArrayBuffer, HashMap}
import scala.collection.JavaConversions._
import scala.language.experimental.macros

import java.util.IdentityHashMap

import chisel3.internal._
import chisel3.internal.Builder._
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.{InstTransform, SourceInfo}
import chisel3.experimental.BaseModule
import _root_.firrtl.annotations.{ModuleName, ModuleTarget, IsModule, IsMember, Named, Target}

object Instance extends SourceInfoDoc {
  /** A wrapper method that all Module instantiations must be wrapped in
    * (necessary to help Chisel track internal state).
    *
    * @param bc the Module being created
    *
    * @return the input module `m` with Chisel metadata properly set
    */
  def apply[T <: BaseModule](bc: T): Instance[T] = macro InstTransform.apply[T]

  /** @group SourceInfoTransformMacro */
  def do_apply[T <: BaseModule](bc: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Instance[T] = {
    //require(bc.isTemplate, "Must pass a template to Instance(..)")
    val ports = experimental.CloneModuleAsRecord(bc)
    Instance(bc, ports, InstanceContext.getContext(Builder.currentModule.get))
  }
}

case class InstanceContext(modulePath: Seq[BaseModule]) {
  val root = modulePath.head
  val path = modulePath.tail.map(x => (x.instanceName, x))
  def descend(mod: BaseModule): InstanceContext = InstanceContext(modulePath :+ mod)
  def ascend(): InstanceContext = InstanceContext(modulePath.dropRight(1))
  def toTarget: IsModule = if(path.isEmpty) root.toTarget else {
    path.foldLeft(root.toTarget: IsModule) { case (tar, (instName, mod)) =>
      tar.instOf(instName, mod.toTarget.module)
    }
  }
}
object InstanceContext {
  def getContext(module: BaseModule): InstanceContext = {
    def contextOf(module: BaseModule): Seq[BaseModule] = {
      module +: module._parent.map( x => contextOf(x) ).getOrElse(Nil)
    }
    val path = contextOf(module)
    InstanceContext(path)
  }
  //def empty = InstanceContext(Nil)
}

case class Instance[T <: BaseModule](template: T, ports: BaseModule.ClonePorts, myContext: InstanceContext) extends NamedComponent {
  override def instanceName = ports.instanceName
  
  val io = ports
  val ioMap = template.getChiselPorts.map(_._2).zip(ports.elements.map(_._2)).toMap
  println(ioMap)
  def apply[X](f: T => X): X = {
    val ret = f(template)
    ret match {
      case x: Data if ioMap.contains(x) => ioMap(x).asInstanceOf[X]
      case _ => ret
    }
  }
}