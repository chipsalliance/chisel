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
  def apply[T <: Data](functionName: String, ret: => T)(clock: Clock, enable: Bool, data: Data*): T = {
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
  def apply[T <: Data](functionName: String, ret: => T)(enable: Bool, data: Data*): T = {
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

// A type util for Chandle.
object DPIType {
  type Chandle = UInt // Can we make this UInt(64)?
  val ChandleType = UInt(64.W)
}

trait DPIFunctionDecl {
  val functionName: String
  val inputs:       Seq[String] // Can we specify types?
  val clocked: Boolean = true
}

// Base trait for a clocked non-void function that returns `T`.
trait DPIClockedNonVoidFunctionDecl[T <: Data] extends DPIFunctionDecl {
  override final val clocked = true
  val retType: T
  def callWithEnable(enable: Bool, data: Data*): T = {
    println(inputs.length)
    println(data.length)
    println(functionName)
    // TODO: Produce a better error message.
    assert(inputs.length == data.length)

    RawClockedNonVoidFunctionCall(functionName, retType)(Module.clock, enable, data: _*)
  }

  def call(data: Data*): T = {
    callWithEnable(true.B, data: _*)
  }
}

// Base trait for a clocked void function.
trait DPIClockedVoidFunctionDecl extends DPIFunctionDecl {
  override final val clocked = true
  def apply(clock: Clock, enable: Bool, data: Data*) = {
    // TODO: Produce a better error message.

    // assert(inputs.length == data.length)
    RawClockedVoidFunctionCall(functionName)(clock, enable, data: _*)
  }
}

abstract class DPIStruct {
  // This holds a pointer to the underlying struct.
  val self: UInt

  // This is set to true if `self` is constructed.
  val initialized = IntrinsicExpr("circt_has_been_reset", Bool())(Module.clock, Module.reset)

  def initialize(constructor: DPIClockedNonVoidFunctionDecl[UInt], args: Data*): UInt = {
    constructor.callWithEnable(Module.reset.asBool && !initialized, args: _*)
  }

  // Call a given function with `self`.
  def callMemberFn[T <: Data](func: DPIClockedNonVoidFunctionDecl[T], default: Option[T] = None)(data: Data*): T = {
    func.callWithEnable(initialized, Seq(self) ++ data: _*)
  }

  def callMemberFn(func: DPIClockedVoidFunctionDecl)(data: Data*) = {
    func(Module.clock, initialized, Seq(self) ++ data: _*)
  }

  // Call DPI function without `initialized` condition. It users' responsibility
  // to call this function under a valid condition.
  def callMemberFnUnsafe[T <: Data](func: DPIClockedNonVoidFunctionDecl[T])(data: Data*): T = {
    func.call(Seq(self) ++ data: _*)
  }

  def callMemberFnUnsafe(func: DPIClockedVoidFunctionDecl)(data: Data*) = {
    func(Module.clock, true.B, Seq(self) ++ data: _*)
  }
}
