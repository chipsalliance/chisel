// See LICENSE for license details.

package chisel3.core

import scala.language.experimental.macros

import chisel3.internal._
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.{SourceInfo}

/** Utility for constructing hardware registers
  *
  * The width of a `Reg` (inferred or not) is copied from the type template
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
  /** Construct a [[Reg]] from a type template with no initialization value (reset is ignored).
    * Value will not change unless the [[Reg]] is given a connection.
    * @param t The template from which to construct this wire
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
  * The two forms of `RegInit` differ in how the type and width of the resulting [[Reg]] are
  * specified.
  *
  * ==Single Argument==
  * The single argument form uses the argument to specify both the type and reset value. For
  * non-literal [[Bits]], the width of the [[Reg]] will be inferred. For literal [[Bits]] and all
  * non-Bits arguments, the type will be copied from the argument. See the following examples for
  * more details:
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
  * The double argument form allows the type of the [[Reg]] and the default connection to be
  * specified independently.
  *
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
  /** Construct a [[Reg]] from a type template initialized to the specified value on reset
    * @param t The type template used to construct this [[Reg]]
    * @param init The value the [[Reg]] is initialized to on reset
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

  /** Construct a [[Reg]] initialized on reset to the specified value.
    * @param init Initial value that serves as a type template and reset value
    */
  def apply[T <: Data](init: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T = {
    val model = (init match {
      // If init is a literal without forced width OR any non-literal, let width be inferred
      case init: Bits if !init.litIsForcedWidth.getOrElse(false) => init.cloneTypeWidth(Width())
      case init => init.cloneTypeFull
    }).asInstanceOf[T]
    RegInit(model, init)
  }

}
