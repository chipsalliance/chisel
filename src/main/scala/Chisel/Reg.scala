// See LICENSE for license details.

package Chisel
import Builder.pushCommand

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
    */
  def apply[T <: Data](t: T = null, next: T = null, init: T = null): T = {
    // REVIEW TODO: rewrite this in a less brittle way, perhaps also in a way
    // that doesn't need two implementations of apply()
    val x = makeType(t, next, init)
    pushCommand(DefRegister(x, Node(x._parent.get.clock), Node(x._parent.get.reset))) // TODO multi-clock
    if (init != null) {
      pushCommand(ConnectInit(x.lref, init.ref))
    }
    if (next != null) {
      x := next
    }
    x
  }

  /** Creates a register without initialization (reset is ignored). Value does
    * not change unless assigned to (using the := operator).
    *
    * @param outType: data type for the register
    */
  def apply[T <: Data](outType: T): T = Reg[T](outType, null.asInstanceOf[T], null.asInstanceOf[T])
}
