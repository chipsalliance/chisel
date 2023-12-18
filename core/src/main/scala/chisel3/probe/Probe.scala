// SPDX-License-Identifier: Apache-2.0

package chisel3.probe

import chisel3._
import chisel3.Data.ProbeInfo
import chisel3.experimental.SourceInfo
import chisel3.internal.{containsProbe, requireIsChiselType, requireNoProbeTypeModifier, Builder}

import scala.language.experimental.macros

class Probe[T <: Data](gen: T)(implicit sourceInfo: SourceInfo) extends Record with OpaqueType{
    private [chisel3.probe] val underlying = ProbeInternal(gen)
    val elements = SeqMap("" -> underlying)
}

class RWProbe[T <: Data](gen: T)(implicit sourceInfo: SourceInfo) extends Record with OpaqueType { 
    private val [chisel3.probe] underlying = RWProbeInternal(gen)
    val elements = SeqMap("" -> underlying)
}

/** Utilities for creating and working with Chisel types that have a probe or
  * writable probe modifier.
  */
private[chisel3] sealed trait ProbeBase {

  protected def apply[T <: Data](source: => T, writable: Boolean)(implicit sourceInfo: SourceInfo): T = {
    val prevId = Builder.idGen.value
    // call Output() to coerce passivity
    val data = Output(source) // should only evaluate source once
    requireNoProbeTypeModifier(data, "Cannot probe a probe.")
    if (containsProbe(data)) {
      Builder.error("Cannot create a probe of an aggregate containing a probe.")
    }
    if (writable && data.isConst) {
      Builder.error("Cannot create a writable probe of a const type.")
    }
    // TODO error if trying to probe a non-passive type
    // https://github.com/chipsalliance/chisel/issues/3609

    val ret: T = if (!data.mustClone(prevId)) data else data.cloneType.asInstanceOf[T]
    setProbeModifier(ret, Some(ProbeInfo(writable)))
    ret
  }
}

object ProbeInternal extends ProbeBase with SourceInfoDoc {

  /** Mark a Chisel type as with a probe modifier.
    */
  private [chisel3] def apply[T <: Data](source: => T): T = macro chisel3.internal.sourceinfo.ProbeTransform.sourceApply[T]

  /** @group SourceInfoTransformMacro */
  private [chisel3] def do_apply[T <: Data](source: => T)(implicit sourceInfo: SourceInfo): T = super.apply(source, false)
}

object RWProbeInternal extends ProbeBase with SourceInfoDoc {

  /** Mark a Chisel type with a writable probe modifier.
    */
  private [chisel3] def apply[T <: Data](source: => T): T = macro chisel3.internal.sourceinfo.ProbeTransform.sourceApply[T]

  /** @group SourceInfoTransformMacro */
  private [chisel3] def do_apply[T <: Data](source: => T)(implicit sourceInfo: SourceInfo): T = super.apply(source, true)
}

object Probe extends SourceInfoDoc {

  /** Mark a Chisel type as with a probe modifier.
    */
  def apply[T <: Data](source: => T): Probe[T] = macro chisel3.internal.sourceinfo.ProbeTransform.sourceApply[T]

  /** @group SourceInfoTransformMacro */
  private [chisel3] def do_apply[T <: Data](source: => T)(implicit sourceInfo: SourceInfo): Probe[T] = new Probe(gen)
}

object RWProbeInternal extends ProbeBase with SourceInfoDoc {

  /** Mark a Chisel type with a writable probe modifier.
    */
  private [chisel3] def apply[T <: Data](source: => T): RWProbe[T] = macro chisel3.internal.sourceinfo.ProbeTransform.sourceApply[T]

  /** @group SourceInfoTransformMacro */
  private [chisel3] def do_apply[T <: Data](source: => T)(implicit sourceInfo: SourceInfo): RWProbe[T] = new RWProbe(gen)
}