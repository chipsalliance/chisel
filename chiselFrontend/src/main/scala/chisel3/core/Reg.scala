// See LICENSE for license details.

package chisel3.core

import scala.language.experimental.macros

import chisel3.internal._
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.{SourceInfo}

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
    val reg = t.chiselCloneType
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
      case next => next.chiselCloneType
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
      case next => next.chiselCloneType
    }).asInstanceOf[T]
    val reg = RegInit(model, init)  // TODO: this makes NO sense

    requireIsHardware(next, "reg next")
    reg := next

    reg
  }
}

object RegInit {
  /** Returns a register pre-initialized (on reset) to the specified value.
    * Register type is inferred from the initializer.
    */
  def apply[T <: Data](init: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T = {
    val model = (init.litArg match {
      // For e.g. Reg(init=UInt(0, k)), fix the Reg's width to k
      case Some(lit) if lit.forcedWidth => init.chiselCloneType
      case _ => init match {
        case init: Bits => init.cloneTypeWidth(Width())
        case init => init.chiselCloneType
      }
    }).asInstanceOf[T]
    RegInit(model, init)
  }

  /** Creates a register given an explicit type and an initialization (reset) value.
    */
  def apply[T <: Data](t: T, init: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T = {
    if (compileOptions.declaredTypeMustBeUnbound) {
      requireIsChiselType(t, "reg type")
    }
    val reg = t.chiselCloneType
    val clock = Node(Builder.forcedClock)
    val reset = Node(Builder.forcedReset)

    reg.bind(RegBinding(Builder.forcedUserModule))
    requireIsHardware(init, "reg initializer")
    pushCommand(DefRegInit(sourceInfo, reg, clock, reset, init.ref))
    reg
  }
}
