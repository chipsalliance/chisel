// SPDX-License-Identifier: Apache-2.0

package chisel3

import scala.collection.immutable.ListMap
import scala.collection.mutable.{ArrayBuffer, HashMap}
import scala.language.experimental.macros

import java.util.IdentityHashMap

import chisel3._
import chisel3.internal._
import chisel3.internal.Builder._
import chisel3.internal.BaseModule.{ModuleClone, InstanceClone, IsClone, InstantiableClone}
import chisel3.internal.sourceinfo.{InstTransform, SourceInfo, SourceInfoTransform}
import chisel3.experimental.BaseModule
import _root_.firrtl.annotations.IsModule

case class Instance[A] private [chisel3] (val cloned: Either[A, IsClone[A]]) {
  def definition: A = cloned match {
    case Left(value: A) => value
    case Right(i: IsClone[A]) => i._proto
  }
  def getInnerDataContext: Option[BaseModule] = cloned match {
    case Left(value: BaseModule)        => Some(value)
    case Left(value: IsInstantiable)    => None
    case Right(i: BaseModule)           => Some(i)
    case Right(i: InstantiableClone[_]) => i._parent
  }
  def getClonedParent: Option[BaseModule] = cloned match {
    case Left(value: BaseModule) => value._parent
    case Right(i: BaseModule)           => i._parent
    case Right(i: InstantiableClone[_]) => i._parent
  }

  private [chisel3] val cache = HashMap[Data, Data]()

  def apply[B, C](that: A => B)(implicit lookup: Lookupable[A, B]): lookup.C = {
    lookup.lookup(that, this)
  }
  def toTarget = cloned match {
    case Left(x: BaseModule) => x.toTarget
    case Right(x: chisel3.internal.BaseModule.ModuleClone[_]) => x.toTarget
    case Right(x: chisel3.internal.BaseModule.InstanceClone[_]) => x.toTarget
    case other => throw new Exception(s"toTarget is not supported on $this")
  }
}
object Instance extends SourceInfoDoc {
  /** A wrapper method that all Module instantiations must be wrapped in
    * (necessary to help Chisel track internal state).
    *
    * @param bc the Module being created
    *
    * @return the input module `m` with Chisel metadata properly set
    */
  def apply[T <: BaseModule, I <: Bundle](bc: Definition[T]): Instance[T] = macro InstTransform.apply[T]

  /** @group SourceInfoTransformMacro */
  def do_apply[T <: BaseModule, I <: Bundle](bc: Definition[T])(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Instance[T] = {
    val ports = experimental.CloneModuleAsRecord(bc.module)
    val clone = ports._parent.get.asInstanceOf[ModuleClone[T]]
    clone._madeFromDefinition = true
    //println(s"In do_apply: ports=$ports")
    new Instance(Right(clone))
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
  
  
  implicit def moduleToInstance[T <: BaseModule] = new Convertable[T, Instance[T]] {
    def convert(that: T): Instance[T] = new Instance(Left(that))
  }
  implicit def isInstantiabletoInstance[T <: IsInstantiable] = new Convertable[T, Instance[T]] {
    def convert(that: T): Instance[T] = new Instance(Left(that))
  }
  implicit def convertSeq[T, R](implicit convertable: Convertable[T, R]) = new Convertable[Seq[T], Seq[R]] {
    def convert(that: Seq[T]): Seq[R] = that.map(convertable.convert)
  }
  //implicit def convertOption[T, R](implicit convertable: Convertable[T, R]) = new Convertable[Option[T], Option[R]] {
  //  def convert(that: Option[T]): Option[R] = that.map(convertable.convert)
  //}

  implicit def convert[T, R](i: T)(implicit convertable: Convertable[T, R]): R = convertable.convert(i)
}

//case class Instance[A] private [chisel3] (name: () => String, template: A, ioMap: Map[Data, Data], context: InstanceContext, ports: Option[BaseModule.ClonePorts]) extends NamedComponent {
//  override def instanceName = ports.map(_.instanceName).getOrElse(name())
//  
//  private [chisel3] val cache = HashMap[Data, Data]()
//
//  def apply[B, C](that: A => B)(implicit lookup: Lookupable[A, B]): lookup.C = {
//    lookup.lookup(that, this)
//  }
//}
