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

object Instance extends SourceInfoDoc {
  /** A wrapper method that all Module instantiations must be wrapped in
    * (necessary to help Chisel track internal state).
    *
    * @param bc the Module being created
    *
    * @return the input module `m` with Chisel metadata properly set
    */
  def apply[T <: BaseModule, I <: Bundle](bc: T): Instance[T] = macro InstTransform.apply[T]

  /** @group SourceInfoTransformMacro */
  def do_apply[T <: BaseModule, I <: Bundle](bc: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Instance[T] = {
    //require(bc.isTemplate, "Must pass a template to Instance(..)")
    val ports = experimental.CloneModuleAsRecord(bc)
    Instance(bc, Some(ports), InstanceContext.getContext(ports._parent.get))
  }
  /*
  implicit class InstanceApplyToInstance[T <: BaseModule](i: Instance[T]) {
    def apply[X <: BaseModule](f: T => Instance[X]): Instance[X] = {
      // If parent instance has no context, derive the context
      val context = i.context.getOrElse(InstanceContext.getContext(i.io._parent.get))
      val handle = f(i.template)
      handle.copy(context = Some(context.descend(handle.io, handle.template)))
    }
  }
  implicit class InstanceApplyToData[T <: BaseModule](i: Instance[T]) {
    def apply[X <: Data](f: T => X): X = {
      // If parent instance has no context, derive the context
      val ret = f(i.template)
      ret match {
        case x: Data if i.ioMap.contains(x) => i.ioMap(x).asInstanceOf[X]
        case x => throwException(s"Cannot return a non-port data type from an instance handle! $x")
      }
    }
  }
  implicit class InstanceApplyToUnit[T <: BaseModule](i: Instance[T]) {
    def apply(f: T => Unit): Unit = {
      // If parent instance has no context, derive the context
      val isEmpty = Builder.instanceContext.isEmpty
      if(isEmpty) Builder.setContext(Some(InstanceContext.getContext(i.io._parent.get)))
      Builder.descend(i.io, i.template)
      val ret = f(i.template)
      Builder.ascend()
      if(isEmpty) Builder.setContext(None)
    }
  }
  */

  
}

case class Instance[T <: BaseModule] private [chisel3] (template: T, ports: Option[BaseModule.ClonePorts], context: InstanceContext) extends NamedComponent {
  override def instanceName = ports.map(_.instanceName).getOrElse(template.instanceName)
  
  private [chisel3] val ioMap = ports match {
    case Some(cp) => template.getChiselPorts.map(_._2).zip(cp.elements.map(_._2)).toMap
    case None => template.getChiselPorts.map(x => x._2 -> x._2).toMap
  }
  private [chisel3] val cache = HashMap[Data, Data]()
  private [chisel3] val descendingContext = context.descend(ports.getOrElse(template), template)

  def module[X <: BaseModule](that: T => X): Instance[X] = macro SourceInfoTransform.thatArg
  def do_module[X <: BaseModule](that: T => X)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Instance[X] = {
    val ret = that(template)
    val inst = Instance(ret, None, descendingContext.descend(InstanceContext.getContext(ret)))
    inst 
  }
  def apply[X](that: T => X): X = macro SourceInfoTransform.thatArg
  def do_apply[X](that: T => X)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): X = {
    val ret = that(template)
    (ret match {
      case x: Data if ioMap.contains(x) => ioMap(x)
      case x: Data if cache.contains(x)=> cache(x)
      case x: Data =>
        val xContext = InstanceContext.getContext(x._parent.get)
        val xmr = XMR.do_apply(x, descendingContext.descend(xContext))
        cache(x) = xmr
        xmr
      case x: Instance[_] =>
        x.copy(context = descendingContext.descend(x.context))
      case x: Unit => x
      case x: BaseModule => throwException(s"Cannot return a module from the apply method, use module(..) instead! $x")
      case x => throwException(s"Cannot return this from an instance handle! $x")
    }).asInstanceOf[X]
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