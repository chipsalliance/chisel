// SPDX-License-Identifier: Apache-2.0

package chisel3

import scala.collection.immutable.ListMap
import scala.collection.mutable.{ArrayBuffer, HashMap}
import scala.language.experimental.macros

import java.util.IdentityHashMap

import chisel3.internal._
import chisel3.internal.Builder._
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.{InstTransform, SourceInfo, SourceInfoTransform}
import chisel3.experimental.BaseModule
import _root_.firrtl.annotations.{ModuleName, ModuleTarget, IsModule, IsMember, Named, Target}
import scala.annotation.implicitNotFound

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
    Instance(bc.module, Some(ports), InstanceContext.getContext(ports._parent.get))
  }
  implicit def toInstance[T <: BaseModule](m: T): Instance[T] = {
    new Instance(m, None, InstanceContext.getContext(m))
  }
}

@implicitNotFound("@interface is only legal when @public is only on subtypes of Data, BaseModule or Instance[_]")
sealed trait Lookupable[A <: BaseModule, -B] {
  type C
  def lookup(that: A => B, ih: Instance[A]): C
}

object Lookupable {
  implicit def lookupModule[A <: BaseModule, B <: BaseModule](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) = new Lookupable[A, B] {
    type C = Instance[B]
    def lookup(that: A => B, ih: Instance[A]): C = {
      import ih._
      val ret = that(template)
      val inst = new Instance(ret, None, descendingContext.descend(InstanceContext.getContext(ret)))
      inst 
    }
  }
  implicit def lookupInstance[A <: BaseModule, B <: BaseModule](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) = new Lookupable[A, Instance[B]] {
    type C = Instance[B]
    def lookup(that: A => Instance[B], ih: Instance[A]): C = {
      import ih._
      val ret = that(template)
      ret.copy(context = descendingContext.descend(ret.context))
    }
  }
  implicit def lookupData[A <: BaseModule, B <: Data](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) = new Lookupable[A, B] {
    type C = B
    def lookup(that: A => B, ih: Instance[A]): C = {
      import ih._
      val ret = that(template)
      ret match {
        case x: Data if ioMap.contains(x) => ioMap(x).asInstanceOf[B]
        case x: Data if cache.contains(x)=> cache(x).asInstanceOf[B]
        case x: Data =>
          val xContext = InstanceContext.getContext(x._parent.get)
          val xmr = XMR.do_apply(x, descendingContext.descend(xContext))
          cache(x) = xmr
          xmr.asInstanceOf[B]
      }
    }
  }
  implicit def lookupOption[A <: BaseModule, B <: Data](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) = new Lookupable[A, Option[B]] {
    type C = Option[B]
    def lookup(that: A => Option[B], ih: Instance[A]): C = {
      import ih._
      val ret = that(template)
      ret.map{ _ match {
          case x: Data if ioMap.contains(x) => ioMap(x).asInstanceOf[B]
          case x: Data if cache.contains(x)=> cache(x).asInstanceOf[B]
          case x: Data =>
            val xContext = InstanceContext.getContext(x._parent.get)
            val xmr = XMR.do_apply(x, descendingContext.descend(xContext))
            cache(x) = xmr
            xmr.asInstanceOf[B]
        }
      }
    }
  }
}

case class Instance[A <: BaseModule] private [chisel3] (template: A, ports: Option[BaseModule.ClonePorts], context: InstanceContext) extends NamedComponent {
  override def instanceName = ports.map(_.instanceName).getOrElse(template.instanceName)
  
  private [chisel3] val ioMap = ports match {
    case Some(cp) => template.getChiselPorts.map(_._2).zip(cp.elements.map(_._2)).toMap
    case None => template.getChiselPorts.map(x => x._2 -> x._2).toMap
  }
  private [chisel3] val cache = HashMap[Data, Data]()
  private [chisel3] val descendingContext: InstanceContext = context.descend(ports.getOrElse(template), template)

  def apply[B, C](that: A => B)(implicit lookup: Lookupable[A, B]): lookup.C = {
    lookup.lookup(that, this)
  }
}






case class InstanceContext(top: BaseModule, instances: Seq[(HasId, BaseModule)]) {
  import InstanceContext._
  def localModule = if(instances.isEmpty) top else instances.last._2
  def descend(instanceName: HasId, module: BaseModule): InstanceContext = {
    val moduleContext = getContext(module)
    InstanceContext(top, instances ++ Seq((instanceName, moduleContext.top)) ++ moduleContext.instances)
  }
  def descend(ic: InstanceContext): InstanceContext = {
    val moduleSeq = top +: (instances.map(_._2))
    val indexOpt = moduleSeq.zipWithIndex.collectFirst {
      case (m, index) if m == ic.top => index
    }
    indexOpt match {
      case Some(0) => ic
      case Some(x) => this.copy(top, instances.slice(0, x) ++ ic.instances)
      case None =>
        require(indexOpt.nonEmpty, s"Descending into ${ic}, but local context is $this")
        ic
    }
  }
  def ascend(): InstanceContext = InstanceContext(top, instances.dropRight(1))
  def toInstanceTarget: IsModule =  {
    instances.foldLeft(top.toTarget: IsModule) { case (im, (instanceName, mod)) =>
      im.instOf(instanceName.getRef.name, mod._component.get.name)
    }
  }
  def toAbsoluteInstanceTarget: IsModule =  {
    instances.foldLeft(top.toAbsoluteTarget: IsModule) { case (im, (instanceName, mod)) =>
      im.instOf(instanceName.getRef.name, mod._component.get.name)
    }
  }
}
object InstanceContext {
  def getContext(module: BaseModule): InstanceContext = {
    module._parent match {
      case None => InstanceContext(module, Nil)
      case Some(parent) if parent == module => InstanceContext(module, Nil)
      case Some(parent) =>
        val ctx = getContext(parent)
        ctx.copy(instances = ctx.instances :+ ((module, module)))
    }
  }
}