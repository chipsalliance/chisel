package chisel3

import chisel3.experimental.{ChiselAnnotation, annotate, requireIsHardware}
import firrtl.transforms.DontTouchAnnotation

/** Marks that a signal should not be removed by Chisel and Firrtl optimization passes
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
  * @note Calling this on [[Data]] creates an annotation that Chisel emits to a separate annotations
  * file. This file must be passed to FIRRTL independently of the `.fir` file. The execute methods
  * in [[chisel3.Driver]] will pass the annotations to FIRRTL automatically.
  */
object dontTouch { // scalastyle:ignore object.name
  /** Marks a signal to be preserved in Chisel and Firrtl
    *
    * @note Requires the argument to be bound to hardware
    * @param data The signal to be marked
    * @return Unmodified signal `data`
    */
  def apply[T <: Data](data: T)(implicit compileOptions: CompileOptions): T = {
    if (compileOptions.checkSynthesizable) {
      requireIsHardware(data, "Data marked dontTouch")
    }
    annotate(new ChiselAnnotation { def toFirrtl = DontTouchAnnotation(data.toNamed) })
    data
  }
}
