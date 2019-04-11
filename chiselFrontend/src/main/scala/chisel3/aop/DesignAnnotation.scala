package chisel3.aop

import chisel3.core.{RawModule, RunFirrtlTransform}

case class DesignAnnotation[T <: RawModule](design: T) extends RunFirrtlTransform with _root_.firrtl.annotations.NoTargetAnnotation {
  override def transformClass: Class[_ <: _root_.firrtl.Transform] = classOf[AspectTransform]
  override def toFirrtl = this
}


