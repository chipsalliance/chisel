// SPDX-License-Identifier: Apache-2.0

package chisel3.simulator

trait ControlAPI {

  /** Enable waveform dumping
    *
    * This control function will enable waveforms from the moment it is applied.
    * A simulator must be compiled with waveform dumping support for this to
    * have an effect.
    *
    * Example usage:
    * {{{
    * enableWaves()
    * }}}
    */
  def enableWaves(): Unit = {
    AnySimulatedModule.current.controller.setTraceEnabled(true)
  }

  /** Enable waveform dumping
    *
    * This control function will disable waveforms from the moment it is
    * applied.  A simulator must be compiled with waveform dumping support for
    * this to have an effect.
    *
    * Example usage:
    * {{{
    * disableWaves()
    * }}}
    */
  def disableWaves(): Unit = {
    AnySimulatedModule.current.controller.setTraceEnabled(false)
  }

}
