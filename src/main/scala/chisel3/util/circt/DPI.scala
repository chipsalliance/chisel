// SPDX-License-Identifier: Apache-2.0

package chisel3.util.circt.dpi

import chisel3._

import chisel3.experimental.{IntParam, IntrinsicModule, Param, StringParam}

private object GetDPIParams {
  def apply(
    functionName: String,
    isClocked:    Boolean,
    inputNames:   Option[Seq[String]],
    outputName:   Option[String] = None
  ): Seq[(String, Param)] = {
    val inputNamesParam =
      inputNames.map(_.mkString(";")).map(x => Seq("inputNames" -> StringParam(x))).getOrElse(Seq())
    val outputNameParam = outputName.map(x => Seq("outputName" -> StringParam(x))).getOrElse(Seq())
    Seq[(String, Param)](
      "functionName" -> functionName,
      "isClocked" -> (if (isClocked) 1 else 0)
    ) ++ inputNamesParam ++ outputNameParam
  }
}

object RawClockedNonVoidFunctionCall {

  /** Creates an intrinsic that calls non-void DPI function at its clock posedge.
    * The result values behave like registers and the DPI function is used as a state
    * transfer function of them.
    *
    * `enable` operand is used to conditionally call the DPI since DPI call could be quite
    * more expensive than native constructs.
    *
    * When an `enable` is false, it means the state transfer function is not called. Hence
    * their values will not be modified in that clock.
    *
    * Please refer https://github.com/llvm/circt/blob/main/docs/Dialects/FIRRTL/FIRRTLIntrinsics.md#dpi-intrinsic-abi for DPI function ABI.
    * @example {{{
    * val a = RawClockedNonVoidFunctionCall("dpi_func_foo", UInt(1.W), clock, enable, b, c)
    * }}}
    */
  def apply[T <: Data](
    functionName: String,
    ret:          => T,
    inputNames:   Option[Seq[String]] = None,
    outputName:   Option[String] = None
  )(clock:        Clock,
    enable:       Bool,
    data:         Data*
  ): T = {
    IntrinsicExpr(
      "circt_dpi_call",
      ret,
      GetDPIParams(functionName, true, inputNames, outputName): _*
    )(
      (Seq(clock, enable) ++ data): _*
    )
  }
}

object RawUnclockedNonVoidFunctionCall {

  /** Creates an intrinsic that calls non-void DPI function for its input value changes.
    * The DPI call is considered as a combinational logic.
    *
    * `enable` operand is used to conditionally call the DPI since DPI call could be quite
    * more expensive than native constructs.
    * When `enable` is false, results of unclocked calls are undefined and evaluated into X.
    *
    * Please refer https://github.com/llvm/circt/blob/main/docs/Dialects/FIRRTL/FIRRTLIntrinsics.md#dpi-intrinsic-abi for DPI function ABI.
    * @example {{{
    * val a = RawUnclockedNonVoidFunctionCall("dpi_func_foo", UInt(1.W), enable, b, c)
    * }}}
    */
  def apply[T <: Data](
    functionName: String,
    ret:          => T,
    inputNames:   Option[Seq[String]] = None,
    outputName:   Option[String] = None
  )(enable:       Bool,
    data:         Data*
  ): T = {
    IntrinsicExpr(
      "circt_dpi_call",
      ret,
      GetDPIParams(functionName, false, inputNames, outputName): _*
    )(
      (Seq(enable) ++ data): _*
    )
  }
}

object RawClockedVoidFunctionCall {

  /** Creates an intrinsic that calls void DPI function at its clock posedge.
    *
    * Please refer https://github.com/llvm/circt/blob/main/docs/Dialects/FIRRTL/FIRRTLIntrinsics.md#dpi-intrinsic-abi for DPI function ABI.
    * @example {{{
    * RawClockedVoidFunctionCall("dpi_func_foo", UInt(1.W), clock, enable, b, c)
    * }}}
    */
  def apply(
    functionName: String,
    inputNames:   Option[Seq[String]] = None
  )(clock:        Clock,
    enable:       Bool,
    data:         Data*
  ): Unit = {
    Intrinsic("circt_dpi_call", GetDPIParams(functionName, true, inputNames): _*)(
      (Seq(clock, enable) ++ data): _*
    )
  }
}

// A common trait for DPI functions.
trait DPIFunctionImport {
  def functionName: String
  def inputNames: Option[Seq[String]] = None
}

trait DPINonVoidFunctionImport[T <: Data] extends DPIFunctionImport {
  /**  Base trait for a non-void function that returns `T`.
    *
    * @tparam T Return type
    * @see Please refer [[https://www.chisel-lang.org/docs/explanations/dpi]] for more detail.
    * @example {{{
    * object Add extends DPINonVoidFunctionImport[UInt] {
    *   override val functionName = "add"
    *   override val ret = UInt(32.W)
    *   override val clocked = false
    *   override val inputNames = Some(Seq("lhs", "rhs"))
    *   override val outputName = Some("result")
    *   final def apply(lhs: UInt, rhs: UInt): UInt = super.call(lhs, rhs)
    * }
    *
    * Add(a, b) // call a native `add` function.
    * }}}
    */
  def ret:     T
  def clocked: Boolean
  def outputName: Option[String] = None
  final def callWithEnable(enable: Bool, data: Data*): T =
    if (clocked) {
      RawClockedNonVoidFunctionCall(functionName, ret, inputNames, outputName)(Module.clock, enable, data: _*)
    } else {
      RawUnclockedNonVoidFunctionCall(functionName, ret, inputNames, outputName)(enable, data: _*)
    }
  final def call(data: Data*): T = callWithEnable(true.B, data: _*)
}

trait DPIVoidFunctionImport extends DPIFunctionImport {
  /**  Base trait for a void function.
    *
    * @see Please refer [[https://www.chisel-lang.org/docs/explanations/dpi]] for more detail.
    * @example {{{
    * object Hello extends DPIVoidFunctionImport {
    *   override val functionName = "hello"
    *   override val clocked = true
    *   final def apply() = super.call()
    * }
    *
    * Hello() // call a native `hello` function.
    * }}}
    */
  def clocked: Boolean
  final def callWithEnable(enable: Bool, data: Data*): Unit =
    RawClockedVoidFunctionCall(functionName, inputNames)(Module.clock, enable, data: _*)
  final def call(data: Data*): Unit = callWithEnable(true.B, data: _*)
}
