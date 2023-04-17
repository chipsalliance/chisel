package chisel3

import chisel3.{Probe, RWProbe}
import chisel3.internal.Builder
import chisel3.internal.firrtl.{ProbeExpr, RWProbeExpr}
import chisel3.experimental.SourceInfo

import scala.language.experimental.macros

private[chisel3] sealed trait ProbeValueBase {
  protected def apply[T <: Data](source: T, writable: Boolean)(implicit sourceInfo: SourceInfo): T = {
    // construct probe to return with cloned info
    val clone = if (writable) RWProbe(source.cloneType) else Probe(source.cloneType)
    clone.bind(chisel3.internal.ProbeBinding(Builder.forcedUserModule, Builder.currentWhen, source))
    if (writable) {
      clone.setRef(RWProbeExpr(source.ref))
    } else {
      clone.setRef(ProbeExpr(source.ref))
    }
    clone
  }
}

object ProbeValue extends ProbeValueBase {

  /** Create a read-only probe expression. */
  def apply[T <: Data](source: T): T = macro chisel3.internal.sourceinfo.ProbeTransform.sourceApply[T]

  def do_apply[T <: Data](source: T)(implicit sourceInfo: SourceInfo): T = super.apply(source, false)
}

object RWProbeValue extends ProbeValueBase {

  /** Create a read/write probe expression. */
  def apply[T <: Data](source: T): T = macro chisel3.internal.sourceinfo.ProbeTransform.sourceApply[T]

  def do_apply[T <: Data](source: T)(implicit sourceInfo: SourceInfo): T = super.apply(source, true)
}
