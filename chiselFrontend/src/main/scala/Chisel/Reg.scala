// See LICENSE for license details.

package Chisel

import internal._
import internal.Builder.pushCommand
import internal.firrtl._
import internal.sourceinfo.{SourceInfo, UnlocatableSourceInfo}

object Reg {
  private[Chisel] def makeType[T <: Data](t: T = null, next: T = null, init: T = null): T = {
    if (t ne null) {
      t.cloneType
    } else if (next ne null) {
      next.cloneTypeWidth(Width())
    } else if (init ne null) {
      init.litArg match {
        // For e.g. Reg(init=UInt(0, k)), fix the Reg's width to k
        case Some(lit) if lit.forcedWidth => init.cloneType
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
  def apply[T <: Data](t: T = null, next: T = null, init: T = null): T =
    // Scala macros can't (yet) handle named or default arguments.
    do_apply(t, next, init)(UnlocatableSourceInfo)

  /** Creates a register without initialization (reset is ignored). Value does
    * not change unless assigned to (using the := operator).
    *
    * @param outType: data type for the register
    */
  def apply[T <: Data](outType: T): T = Reg[T](outType, null.asInstanceOf[T], null.asInstanceOf[T])

  def do_apply[T <: Data](t: T, next: T, init: T)(implicit sourceInfo: SourceInfo): T = {
    // TODO: write this in a way that doesn't need nulls (bad Scala style),
    // null.asInstanceOf[T], and two constructors. Using Option types are an
    // option, but introduces cumbersome syntax (wrap everything in a Some()).
    // Implicit conversions to Option (or similar) types were also considered,
    // but Scala's type inferencer and implicit insertion isn't smart enough
    // to resolve all use cases. If the type inferencer / implicit resolution
    // system improves, this may be changed.
    val x = makeType(t, next, init)
    val clock = Node(x._parent.get.clock) // TODO multi-clock
    if (init == null) {
      pushCommand(DefReg(sourceInfo, x, clock))
    } else {
      pushCommand(DefRegInit(sourceInfo, x, clock, Node(x._parent.get.reset), init.ref))
    }
    if (next != null) {
      x := next
    }
    x
  }
}
