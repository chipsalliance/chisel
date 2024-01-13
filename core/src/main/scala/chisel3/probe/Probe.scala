// SPDX-License-Identifier: Apache-2.0

package chisel3.probe

import chisel3._
import chisel3.Data.ProbeInfo
import chisel3.experimental.{OpaqueType, SourceInfo}
import chisel3.internal.{containsProbe, requireIsChiselType, requireNoProbeTypeModifier, Builder}

import scala.collection.immutable.SeqMap
import scala.language.experimental.macros

abstract sealed trait ProbeLike[+T <: Data] extends Record with OpaqueType {
  private[chisel3] def underlying: T
}

class Probe[+T <: Data](gen: => T)(implicit sourceInfo: SourceInfo) extends ProbeLike[T] {
  private[chisel3] val underlying = ProbeInternal(gen, false)
  val elements = SeqMap("" -> underlying)
}

class RWProbe[+T <: Data](gen: => T)(implicit sourceInfo: SourceInfo) extends ProbeLike[T] {
  private[chisel3] val underlying = ProbeInternal(gen, true)
  val elements = SeqMap("" -> underlying)
}

/** Utilities for creating and working with Chisel types that have a probe or
  * writable probe modifier.
  */
private[probe] object ProbeInternal {

  private[probe] def apply[T <: Data](source: => T, writable: Boolean)(implicit sourceInfo: SourceInfo): T = {
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

object Probe extends SourceInfoDoc {

  /** Mark a Chisel type as with a probe modifier.
    */
  def apply[T <: Data](source: => T): Probe[T] = macro chisel3.internal.sourceinfo.ProbeTransform.sourceApply[T]

  /** @group SourceInfoTransformMacro */
  def do_apply[T <: Data](source: => T)(implicit sourceInfo: SourceInfo): Probe[T] = new Probe(source)
}

object RWProbe extends SourceInfoDoc {

  /** Mark a Chisel type with a writable probe modifier.
    */
  def apply[T <: Data](source: => T): RWProbe[T] = macro chisel3.internal.sourceinfo.ProbeTransform.sourceApply[T]

  /** @group SourceInfoTransformMacro */
  def do_apply[T <: Data](source: => T)(implicit sourceInfo: SourceInfo): RWProbe[T] = new RWProbe(source)
}
