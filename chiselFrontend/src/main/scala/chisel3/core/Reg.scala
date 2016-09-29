// See LICENSE for license details.

package chisel3.core

import chisel3.internal._
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.{SourceInfo, UnlocatableSourceInfo}
import chisel3.ImplicitCompileOptions

object Reg {
  private[core] def makeType[T <: Data](compileOptions: ImplicitCompileOptions, t: T = null, next: T = null, init: T = null): T = {
    if (t ne null) {
      if (compileOptions.declaredTypeMustBeUnbound) {
        Binding.checkUnbound(t, s"t ($t) must be unbound Type. Try using cloneType?")
      }
      t.chiselCloneType
    } else if (next ne null) {
      next.cloneTypeWidth(Width())
    } else if (init ne null) {
      init.litArg match {
        // For e.g. Reg(init=UInt(0, k)), fix the Reg's width to k
        case Some(lit) if lit.forcedWidth => init.chiselCloneType
        case _ => init.cloneTypeWidth(Width())
      }
    } else {
      throwException("cannot infer type")
    }
  }

  /** Creates a register with optional next and initialization values.
    *
    * @param t: data type for the register
    * @param next: new value register is to be updated with every cycle (or
    * empty to not update unless assigned to using the := operator)
    * @param init: initialization value on reset (or empty for uninitialized,
    * where the register value persists across a reset)
    *
    * @note this may result in a type error if called from a type parameterized
    * function, since the Scala compiler isn't smart enough to know that null
    * is a valid value. In those cases, you can either use the outType only Reg
    * constructor or pass in `null.asInstanceOf[T]`.
    */
  def apply[T <: Data](t: T = null, next: T = null, init: T = null)(implicit sourceInfo: SourceInfo, compileOptions: ImplicitCompileOptions): T =
    // Scala macros can't (yet) handle named or default arguments.
    do_apply(t, next, init)(sourceInfo, compileOptions)

  /** Creates a register without initialization (reset is ignored). Value does
    * not change unless assigned to (using the := operator).
    *
    * @param outType: data type for the register
    */
  def apply[T <: Data](outType: T)(implicit sourceInfo: SourceInfo, compileOptions: ImplicitCompileOptions): T = Reg[T](outType, null.asInstanceOf[T], null.asInstanceOf[T])(sourceInfo, compileOptions)

  def do_apply[T <: Data](t: T, next: T, init: T)(implicit sourceInfo: SourceInfo, compileOptions: ImplicitCompileOptions = chisel3.ExplicitCompileOptions.NotStrict): T = {
    // TODO: write this in a way that doesn't need nulls (bad Scala style),
    // null.asInstanceOf[T], and two constructors. Using Option types are an
    // option, but introduces cumbersome syntax (wrap everything in a Some()).
    // Implicit conversions to Option (or similar) types were also considered,
    // but Scala's type inferencer and implicit insertion isn't smart enough
    // to resolve all use cases. If the type inferencer / implicit resolution
    // system improves, this may be changed.
    val x = makeType(compileOptions, t, next, init)
    val clock = Node(x._parent.get.clock) // TODO multi-clock

    // Bind each element of x to being a Reg
    Binding.bind(x, RegBinder(Builder.forcedModule), "Error: t")

    if (init == null) {
      pushCommand(DefReg(sourceInfo, x, clock))
    } else {
      Binding.checkSynthesizable(init, s"'init' ($init)")
      pushCommand(DefRegInit(sourceInfo, x, clock, Node(x._parent.get.reset), init.ref))
    }
    if (next != null) {
      Binding.checkSynthesizable(next, s"'next' ($next)")
      x := next
    }
    x
  }
}
