package chisel3

import chisel3._
import chisel3.experimental.BaseModule
import chisel3.internal._
import _root_.firrtl.annotations.{ModuleName, ModuleTarget, IsModule, IsMember, Named, Target}

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
  def empty = InstanceContext(Builder.currentModule.get, Nil)
}