// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.{annotate, requireIsHardware, ChiselAnnotation}
import firrtl.transforms.DontTouchAnnotation

/** Marks that a signal's leaves are an optimization barrier to Chisel and the 
  *  FIRRTL compiler. This has the effect of guaranteeing that a signal's leaves
  *  will not be removed.
  *
  * @example {{{
  * class MyModule extends Module {
  *   val io = IO(new Bundle {
  *     val a = Input(new Bundle {val a1: UInt(32.W)), val a2: UInt(32.W)})
  *     val b = Output(new Bundle {val a1: UInt(32.W)), val a2: UInt(32.W)})
  *   })
  *   io.b := io.a
  *   // This will preserve io.b.b1 and io.b.b2, though not necessarily as a 
  *   // verilog bundle, they may be individual named entities.
  *   dontTouchLeaves(io.b)
  * }
  * }}}
  * @note Because this is an optimization barrier, constants will not be propagated through a signal marked as
  * dontTouch.
  */
object dontTouchLeaves {

  /** Mark a signal's leaves as an optimization barrier to Chisel and FIRRTL.
    *
    * @note Requires the argument to be bound to hardware
    * @param data The signal to be marked
    * @return Unmodified signal `data`
    */
  def apply[T <: Data](data: T): T = {
    requireIsHardware(data, "Data marked dontTouchLeaves")
    data match {
       case agg: Aggregate => agg.getElements.foreach(dontTouchLeaves.apply)
       case elt: Element   => annotate(new ChiselAnnotation { def toFirrtl = DontTouchAnnotation(data.toNamed) })
       case _              => throw new ChiselException("Non-hardware dontTouchLeaves")
    }
    data
  }
}
