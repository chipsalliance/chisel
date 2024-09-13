// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.internal._
import chisel3.internal.Builder.pushOp
import chisel3.experimental.{requireIsHardware, SourceInfo}
import chisel3.internal.firrtl.ir._
import chisel3.internal.firrtl.ir.PrimOp._

private[chisel3] trait MuxImpl {

  protected def _applyImpl[T <: Data](
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
