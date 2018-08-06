package chisel3.libs.aspect

import chisel3.experimental.{BaseModule, MultiIOModule, dontTouch}
import chisel3.{Data, Input, Output, chiselTypeOf}
import firrtl.annotations.Component

import scala.collection.mutable


class MonitorModule(val instName: String, val parent: BaseModule) extends MultiIOModule with Aspect {
  //override val instanceName = instName
  val ios = mutable.ArrayBuffer[Data]()
  val cmrData = mutable.HashMap[Data, Data]()
  val cmrComponent = mutable.HashMap[Component, ( () => Component)]()
  val refs = mutable.HashSet[Data]()
  var result: Option[Data] = None
  def instComponent = parent.toNamed.inst(instName).of(name)
  def addInput[D<:Data](ref: D): D = {
    if(!refs.contains(ref)) {
      val x = IO(Input(chiselTypeOf(ref)))
      ios += x
      cmrData += ((ref, x))
      cmrComponent += ((ref.toNamed, { () => instComponent.ref(x.instanceName) } ))
      refs += ref
      x
    } else cmrData(ref).asInstanceOf[D]
  }
  def addOutput[D<:Data](ref: D): D = {
    if(!refs.contains(ref)) {
      val x = IO(Output(chiselTypeOf(ref)))
      ios += x
      cmrData += ((ref, x))
      cmrComponent += ((ref.toNamed, { () => instComponent.ref(x.instanceName) } ))
      refs += ref
      x
    } else cmrData(ref).asInstanceOf[D]
  }
  def snip[M<:MultiIOModule, D<:Data](s: Snippet[M, D]): MonitorModule = {
    Aspect.withAspect(this) {
      val flag = s.snip(parent.asInstanceOf[M])
      result = Some(flag)
      dontTouch(flag)
    }
    this
  }
}

