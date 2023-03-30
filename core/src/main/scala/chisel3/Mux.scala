// SPDX-License-Identifier: Apache-2.0

package chisel3

import scala.language.experimental.macros

import chisel3.internal._
import chisel3.internal.Builder.pushOp
import chisel3.experimental.{requireIsHardware, SourceInfo}
import chisel3.internal.sourceinfo.MuxTransform
import chisel3.internal.firrtl._
import chisel3.internal.firrtl.PrimOp._

object Mux extends SourceInfoDoc {

  /** Creates a mux, whose output is one of the inputs depending on the
    * value of the condition.
    *
    * @param cond condition determining the input to choose
    * @param con the value chosen when `cond` is true
    * @param alt the value chosen when `cond` is false
    * @example
    * {{{
    * val muxOut = Mux(data_in === 3.U, 3.U(4.W), 0.U(4.W))
    * }}}
    */
  def apply[T <: Data](cond: Bool, con: T, alt: T): T = macro MuxTransform.apply[T]

  /** @group SourceInfoTransformMacro */
  def do_apply[T <: Data](
    cond: Bool,
    con:  T,
    alt:  T
  )(
    implicit sourceInfo: SourceInfo
  ): T = {
    requireIsHardware(cond, "mux condition")
    requireIsHardware(con, "mux true value")
    requireIsHardware(alt, "mux false value")
    val d = cloneSupertype(Seq(con, alt), "Mux")
    val conRef = con match { // this matches chisel semantics (DontCare as object) to firrtl semantics (invalidate)
      case DontCare =>
        val dcWire = Wire(d)
        dcWire := DontCare
        dcWire.ref
      case _ => con.ref
    }
    val altRef = alt match {
      case DontCare =>
        val dcWire = Wire(d)
        dcWire := DontCare
        dcWire.ref
      case _ => alt.ref
    }
    pushOp(DefPrim(sourceInfo, d, MultiplexOp, cond.ref, conRef, altRef))
  }
}
