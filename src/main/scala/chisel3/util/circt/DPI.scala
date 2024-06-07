// SPDX-License-Identifier: Apache-2.0

package chisel3.util.circt.dpi

import chisel3._
object RawClockedNonVoidFunctionCall {

  /** Creates an intrinsic that calls non-void DPI function at its clock posedge.
    *
    * @example {{{
    * val a = RawClockedNonVoidFunctionCall("dpi_func_foo", UInt(1.W), clock, enable, b, c)
    * }}}
    */
  def apply[T <: Data](functionName: String, ret: => T)(clock: Clock, enable: Bool, data: Data*): Data = {
    IntrinsicExpr("circt_dpi_call", ret, "functionName" -> functionName, "isClocked" -> 1)(
      (Seq(clock, enable) ++ data): _*
    )
  }
}

object RawUnlockedNonVoidFunctionCall {

  /** Creates an intrinsic that calls non-void DPI function when input values are changed.
    *
    * @example {{{
    * val a = RawUnlockedNonVoidFunctionCall("dpi_func_foo", UInt(1.W), enable, b, c)
    * }}}
    */
  def apply[T <: Data](functionName: String, ret: => T)(enable: Bool, data: Data*): Data = {
    IntrinsicExpr("circt_dpi_call", ret, "functionName" -> functionName, "isClocked" -> 0)(
      (Seq(enable) ++ data): _*
    )
  }
}

object RawClockedVoidFunctionCall {

  /** Creates an intrinsic that calls void DPI function at its clock posedge.
    *
    * @example {{{
    * RawClockedVoidFunctionCall("dpi_func_foo", UInt(1.W), clock, enable, b, c)
    * }}}
    */
  def apply(functionName: String)(clock: Clock, enable: Bool, data: Data*) = {
    Intrinsic("circt_dpi_call", "functionName" -> functionName, "isClocked" -> 1)(
      (Seq(clock, enable) ++ data): _*
    )
  }
}
