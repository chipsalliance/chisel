package chisel3.aop

import chisel3.core.{RawModule, RunFirrtlTransform}
import firrtl.annotations.NoTargetAnnotation
import firrtl.Transform

/** Contains the top-level elaborated Chisel design.
  *
  * By default is created during Chisel elaboration and passed to the FIRRTL compiler.
  * @param design top-level Chisel design
  * @tparam DUT Type of the top-level Chisel design
  */
case class DesignAnnotation[DUT <: RawModule](design: DUT) extends RunFirrtlTransform with NoTargetAnnotation {
  override def transformClass: Class[_ <: Transform] = classOf[Transform]
  override def toFirrtl: DesignAnnotation[DUT] = this
}
