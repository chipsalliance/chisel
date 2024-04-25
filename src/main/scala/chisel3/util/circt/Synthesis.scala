// SPDX-License-Identifier: Apache-2.0

package chisel3.util.circt

import chisel3._
import chisel3.experimental.{requireIsHardware, SourceInfo}
import chisel3.internal.Builder

/** Utility for constructing 2-to-1 MUX cell intrinsic. This intrinsic is lowered into verilog
  * with vendor specic pragmas that guarantee utilization of 2-to-1 MUX cell in the synthesis process.
  * Semantically `Mux2Cell(cond, con, alt)` is equivalent to `Mux(cond, con, alt)` for all `cond`, `con` and `alt`.
  */
object Mux2Cell {

  /**
    * Creates an intrinsic which will be lowered to a 2-to-1 MUX cell in synthesis tools.
    * @param cond condition determining the input to choose
    * @param con the value chosen when `cond` is true
    * @param alt the value chosen when `cond` is false
    * @example
    * {{{
    * val muxOut = Mux2Cell(data_in === 3.U, 3.U(4.W), 0.U(4.W))
    * }}}
    */
  def apply[T <: Data](cond: UInt, con: T, alt: T)(implicit sourceInfo: SourceInfo): T = {
    requireIsHardware(cond, "MUX2 cell selector")
    requireIsHardware(con, "MUX2 cell true value")
    requireIsHardware(alt, "MUX2 cell false value")
    val d = cloneSupertype(Seq(con, alt), "Mux2Cell")
    IntrinsicExpr("circt_mux2cell", d)(cond, con, alt)
  }
}

/** Utility for constructing 4-to-1 MUX cell intrinsic. This intrinsic is lowered into verilog
  * with vendor specic pragmas that guarantee utilization of 4-to-1 MUX cell in the synthesis process.
  */
object Mux4Cell {

  /**
    * Creates an intrinsic which will be lowered to a 4-to-1 MUX cell in synthesis tools.
    * @param sel 2-bit unsigned integer determining the input to choose.
    * @param v3 the value chosen when `selector` is `3`
    * @param v2 the value chosen when `selector` is `2`
    * @param v1 the value chosen when `selector` is `1`
    * @param v0 the value chosen when `selector` is `0`
    * @example {{{
    * v := Mux4Cell(sel, v3, v2, v1, v0)
    * }}}
    */
  def apply[T <: Data](sel: UInt, v3: T, v2: T, v1: T, v0: T)(implicit sourceInfo: SourceInfo): T = {
    requireIsHardware(sel, "4-to-1 mux selector")
    requireIsHardware(v3, "MUX4 cell input value when selector == 3")
    requireIsHardware(v2, "MUX4 cell input value when selector == 2")
    requireIsHardware(v1, "MUX4 cell input value when selector == 1")
    requireIsHardware(v0, "MUX4 cell input value when selector == 0")
    val d = cloneSupertype(Seq(v3, v2, v1, v0), "Mux4Cell")
    IntrinsicExpr("circt_mux4cell", d)(sel, v3, v2, v1, v0)
  }
}
