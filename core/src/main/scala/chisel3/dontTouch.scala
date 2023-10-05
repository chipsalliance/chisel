// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.{annotate, requireIsHardware, ChiselAnnotation}
import firrtl.transforms.DontTouchAnnotation

/** Marks that a signal is an optimization barrier to Chisel and the FIRRTL compiler. This has the effect of
  * guaranteeing that a signal will not be removed.
  *
  * @example {{{
  * class MyModule extends Module {
  *   val io = IO(new Bundle {
  *     val a = Input(UInt(32.W))
  *     val b = Output(UInt(32.W))
  *   })
  *   io.b := io.a
  *   val dead = io.a +% 1.U // normally dead would be pruned by DCE
  *   dontTouch(dead) // Marking it as such will preserve it
  * }
  * }}}
  * @note Because this is an optimization barrier, constants will not be propagated through a signal marked as
  * dontTouch.
  */
object dontTouch {

  /** Mark a signal as an optimization barrier to Chisel and FIRRTL.
    *
    * @note Requires the argument to be bound to hardware
    * @param data The signal to be marked
    * @return Unmodified signal `data`
    */
  def apply[T <: Data](data: T): T = {
    requireIsHardware(data, "Data marked dontTouch")
    annotate(new ChiselAnnotation { def toFirrtl = DontTouchAnnotation(data.toNamed) })
    data
  }
}
