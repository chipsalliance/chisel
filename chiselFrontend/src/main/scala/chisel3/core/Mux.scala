// See LICENSE for license details.

package chisel3.core

import scala.language.experimental.macros

import chisel3.internal._
import chisel3.internal.Builder.{pushOp}
import chisel3.internal.sourceinfo.{SourceInfo, MuxTransform}
import chisel3.internal.firrtl._
import chisel3.internal.firrtl.PrimOp._

object Mux {
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

  def do_apply[T <: Data](cond: Bool, con: T, alt: T)(implicit sourceInfo: SourceInfo,
      compileOptions: CompileOptions): T = {
    requireIsHardware(cond, "mux condition")
    requireIsHardware(con, "mux true value")
    requireIsHardware(alt, "mux false value")
    val d = cloneSupertype(Seq(con, alt), "Mux")
    pushOp(DefPrim(sourceInfo, d, MultiplexOp, cond.ref, con.ref, alt.ref))
  }
}
