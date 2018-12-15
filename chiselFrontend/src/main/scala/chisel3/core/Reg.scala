// See LICENSE for license details.

package chisel3.core

import scala.language.experimental.macros

import chisel3.internal._
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.{SourceInfo}

/** Utility for constructing hardware registers
  *
  * =Width Inference=
  * The width of a `Reg` (inferred or not) will be copied from the type template
  * {{{
  * val r0 = Reg(UInt()) // width is inferred
  * val r1 = Reg(UInt(8.W)) // width is set to 8
  *
  * val r2 = Reg(Vec(4, UInt())) // width is inferred
  * val r3 = Reg(Vec(4, UInt(8.W))) // width of each element is set to 8
  *
  * class MyBundle {
  *   val unknown = UInt()
  *   val known   = UInt(8.W)
  * }
  * val r4 = Reg(new MyBundle)
  * // Width of r4.unknown is inferred
  * // Width of r4.known is set to 8
  * }}}
  *
  */
object Reg {
  /** Creates a register without initialization (reset is ignored). Value does
    * not change unless assigned to (using the := operator).
    *
    * @param t: data type for the register
    */
  def apply[T <: Data](t: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T = {
    if (compileOptions.declaredTypeMustBeUnbound) {
      requireIsChiselType(t, "reg type")
    }
    val reg = t.cloneTypeFull
    val clock = Node(Builder.forcedClock)

    reg.bind(RegBinding(Builder.forcedUserModule))
    pushCommand(DefReg(sourceInfo, reg, clock))
    reg
  }
}

object RegNext {
  /** Returns a register with the specified next and no reset initialization.
    *
    * Essentially a 1-cycle delayed version of the input signal.
    */
  def apply[T <: Data](next: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T = {
    val model = (next match {
      case next: Bits => next.cloneTypeWidth(Width())
      case next => next.cloneTypeFull
    }).asInstanceOf[T]
    val reg = Reg(model)

    requireIsHardware(next, "reg next")
    reg := next

    reg
  }

  /** Returns a register with the specified next and reset initialization.
    *
    * Essentially a 1-cycle delayed version of the input signal.
    */
  def apply[T <: Data](next: T, init: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T = {
    val model = (next match {
      case next: Bits => next.cloneTypeWidth(Width())
      case next => next.cloneTypeFull
    }).asInstanceOf[T]
    val reg = RegInit(model, init)  // TODO: this makes NO sense

    requireIsHardware(next, "reg next")
    reg := next

    reg
  }
}

/** Utility for constructing hardware registers with an initialization value.
  *
  * The register is set to the initialization value when the current implicit `reset` is high
  *
  * =Width Inference=
  *
  * The two forms of `RegInit` have differing width inference semantics:
  *
  * ==Single Argument==
  * There are 4 cases of type inference for single argument `RegInit`:
  *
  * 1. Literal [[Bits]] initializer: width will be set to match
  * {{{
  * val r1 = RegInit(1.U) // width will be inferred to be 1
  * val r2 = RegInit(1.U(8.W)) // width is set to 8
  * }}}
  *
  * 2. Non-Literal [[Element]] initializer - width will be inferred
  * {{{
  * val x = Wire(UInt())
  * val y = Wire(UInt(8.W))
  * val r1 = RegInit(x) // width will be inferred
  * val r2 = RegInit(y) // width will be inferred
  * }}}
  *
  * 3. [[Aggregate]] initializer - width will be set to match the aggregate
  *
  * {{{
  * class MyBundle {
  *   val unknown = UInt()
  *   val known   = UInt(8.W)
  * }
  * val w1 = Reg(new MyBundle)
  * val w2 = RegInit(w1)
  * // Width of w2.unknown is inferred
  * // Width of w2.known is set to 8
  * }}}
  *
  * ==Double Argument==
  * The width inference semantics for `RegInit` with two arguments match those of [[Reg]]. The
  * first argument to `RegInit` is the type template which defines the width of the `Reg` in
  * exactly the same way as the only argument to [[Wire]].
  *
  * More explicitly, you can reason about `RegInit` with multiple arguments as if it were defined
  * as:
  * {{{
  * def RegInit[T <: Data](t: T, init: T): T = {
  *   val x = Reg(t)
  *   x := init
  *   x
  * }
  * }}}
  */
object RegInit {
  /** Returns a register pre-initialized (on reset) to the specified value.
    * Register type is inferred from the initializer.
    */
  def apply[T <: Data](init: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T = {
    val model = (init match {
      // If init is a literal without forced width OR any non-literal, let width be inferred
      case init: Bits if !init.litIsForcedWidth.getOrElse(false) => init.cloneTypeWidth(Width())
      case init => init.cloneTypeFull
    }).asInstanceOf[T]
    RegInit(model, init)
  }

  /** Creates a register given an explicit type and an initialization (reset) value.
    */
  def apply[T <: Data](t: T, init: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T = {
    if (compileOptions.declaredTypeMustBeUnbound) {
      requireIsChiselType(t, "reg type")
    }
    val reg = t.cloneTypeFull
    val clock = Builder.forcedClock.ref
    val reset = Builder.forcedReset.ref

    reg.bind(RegBinding(Builder.forcedUserModule))
    requireIsHardware(init, "reg initializer")
    pushCommand(DefRegInit(sourceInfo, reg, clock, reset, init.ref))
    reg
  }
}
