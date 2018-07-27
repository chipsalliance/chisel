package chisel3.libs.aspect

import chisel3.experimental.{BaseModule, MultiIOModule, dontTouch}
import chisel3.{Data, Input, Output, chiselTypeOf}

import scala.collection.mutable


class MonitorModule(val instName: String, val parent: BaseModule) extends MultiIOModule with Aspect {
  //override val instanceName = instName
  val ios = mutable.ArrayBuffer[Data]()
  val cmrs = mutable.HashMap[Data, Data]()
  val refs = mutable.HashSet[Data]()
  var result: Option[Data] = None
  def addInput[D<:Data](ref: D): D = {
    if(!refs.contains(ref)) {
      val x = IO(Input(chiselTypeOf(ref)))
      ios += x
      cmrs += ((ref, x))
      refs += ref
      x
    } else cmrs(ref).asInstanceOf[D]
  }
  def addOutput[D<:Data](ref: D): D = {
    if(!refs.contains(ref)) {
      val x = IO(Output(chiselTypeOf(ref)))
      ios += x
      cmrs += ((ref, x))
      refs += ref
      x
    } else cmrs(ref).asInstanceOf[D]
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

