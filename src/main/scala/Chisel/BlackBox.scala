// See LICENSE for license details.

package Chisel

/** Defines a black box, which is a module that can be referenced from within
  * Chisel, but is not defined in the emitted Verilog. Useful for connecting
  * to RTL modules defined outside Chisel.
  *
  * @example
  * {{{
  * class DSP48E1 extends BlackBox {
  *   val io = new Bundle // Create I/O with same as DSP
  *   val dspParams = new VerilogParameters // Create Parameters to be specified
  *   setVerilogParams(dspParams)
  *   // Implement functionality of DSP to allow simulation verification
  * }
  * }}}
  */
// TODO: actually implement BlackBox (this hack just allows them to compile)
// REVIEW TODO: make Verilog parameters part of the constructor interface?
abstract class BlackBox(_clock: Clock = null, _reset: Bool = null) extends Module(_clock = _clock, _reset = _reset) {
  def setVerilogParameters(s: String): Unit = {}
}
