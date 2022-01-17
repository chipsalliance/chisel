// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.{annotate, ChiselAnnotation}
import chisel3.internal.NamedComponent
import firrtl.annotations.PresetAnnotation

private class ChiselResetIsInitialValueAnnotation(reset: NamedComponent) extends ChiselAnnotation {

  /** Conversion to FIRRTL Annotation */
  override def toFirrtl = PresetAnnotation(reset.toTarget)
}

/** Forces the module `reset` to be an initial value style reset,
  * i.e. all registers will take on their reset value
  * when the simulation is started or when the FPGA is programmed.
  *
  * @note this kind of reset commonly does not work for ASICs!
  */
trait RequireResetIsInitialValue extends RequireAsyncReset {
  annotate(new ChiselResetIsInitialValueAnnotation(reset))
}

object withResetIsInitialValue {

  /** Creates a new Reset scope with an initial value style reset,
    * i.e. all registers will take on their reset value
    * when the simulation is started or when the FPGA is programmed.
    *
    * @note this kind of reset commonly does not work for ASICs!
    */
  def apply[T](block: => T): T = {
    val init = WireInit(0.B.asAsyncReset)
    annotate(new ChiselResetIsInitialValueAnnotation(init))
    withReset(init)(block)
  }
}
