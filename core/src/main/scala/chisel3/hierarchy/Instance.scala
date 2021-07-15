// SPDX-License-Identifier: Apache-2.0

package chisel3

import scala.collection.immutable.ListMap
import scala.collection.mutable.{ArrayBuffer, HashMap}
import scala.language.experimental.macros

import java.util.IdentityHashMap

import chisel3._
import chisel3.internal._
import chisel3.internal.Builder._
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.{InstTransform, SourceInfo, SourceInfoTransform}
import chisel3.experimental.BaseModule

object Instance extends SourceInfoDoc {
  /** A wrapper method that all Module instantiations must be wrapped in
    * (necessary to help Chisel track internal state).
    *
    * @param bc the Module being created
    *
    * @return the input module `m` with Chisel metadata properly set
    */
  def apply[T <: BaseModule, I <: Bundle](bc: Template[T]): Instance[T] = macro InstTransform.apply[T]

  /** @group SourceInfoTransformMacro */
  def do_apply[T <: BaseModule, I <: Bundle](bc: Template[T])(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Instance[T] = {
    //require(bc.isTemplate, "Must pass a template to Instance(..)")
    val ports = experimental.CloneModuleAsRecord(bc.module)
    Instance(() => ports.instanceName, bc.module, portMap(ports, bc.module), InstanceContext.getContext(ports._parent.get).descend(ports, bc.module), Some(ports))
  }

  def portMap(m: BaseModule): Map[Data, Data] = {
    m.getChiselPorts.map{ x => x._2 -> x._2 }.toMap
  }
  def portMap(ports: BaseModule.ClonePorts, m: BaseModule): Map[Data, Data] = {
    val name2Port = ports.elements
    m.getChiselPorts.map { case (name, data) => data -> name2Port(name) }.toMap
    //m.getChiselPorts.map{_._2).zip(ports.elements.map(_._2)).toMap
  }
  import scala.language.implicitConversions
  //implicit def convertSeq[T <: IsInstantiable](i: T): Instance[T] = {
  //  i match {
  //    case m: BaseModule => new Instance(() => m.instanceName, i, portMap(m), InstanceContext.getContext(m).descend(m, m), None)
  //    case _ => new Instance(() => "", i, Map.empty, InstanceContext.empty, None)
  //  }
  //}
  sealed trait Convertable[-A, +B] {
    def convert(that: A): B
  }
  
  
  implicit def isInstantiabletoInstance[T <: IsInstantiable] = new Convertable[T, Instance[T]] {
    def convert(that: T): Instance[T] = {
      that match {
        case m: BaseModule => new Instance(() => m.instanceName, that, portMap(m), InstanceContext.getContext(m).descend(m, m), None)
        case _ => new Instance(() => "", that, Map.empty, InstanceContext.empty, None)
      }
    }
  }
  implicit def convertSeq[T, R](implicit convertable: Convertable[T, R]) = new Convertable[Seq[T], Seq[R]] {
    def convert(that: Seq[T]): Seq[R] = that.map(convertable.convert)
  }
  //implicit def convertOption[T, R](implicit convertable: Convertable[T, R]) = new Convertable[Option[T], Option[R]] {
  //  def convert(that: Option[T]): Option[R] = that.map(convertable.convert)
  //}

  implicit def convert[T, R](i: T)(implicit convertable: Convertable[T, R]): R = convertable.convert(i)
}

case class Instance[A] private [chisel3] (name: () => String, template: A, ioMap: Map[Data, Data], context: InstanceContext, ports: Option[BaseModule.ClonePorts]) extends NamedComponent {
  override def instanceName = ports.map(_.instanceName).getOrElse(name())
  
  private [chisel3] val cache = HashMap[Data, Data]()

  def apply[B, C](that: A => B)(implicit lookup: Lookupable[A, B]): lookup.C = {
    lookup.lookup(that, this)
  }
}
