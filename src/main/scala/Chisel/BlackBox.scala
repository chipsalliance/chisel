// See LICENSE for license details.

package Chisel

/** Defines a black box, which is a module that can be referenced from within
  * Chisel, but is not defined in the emitted Verilog. Useful for connecting
  * to RTL modules defined outside Chisel.
  *
  * @example
  * {{{
  * ... to be written once a spec is finalized ...
  * }}}
  */
// REVIEW TODO: make Verilog parameters part of the constructor interface?
abstract class BlackBox(_clock: Clock = null, _reset: Bool = null)
    extends Module(_clock = _clock, _reset = _reset) {
  // TODO: actually implement this.
  def setVerilogParameters(s: String): Unit = {}

  // The body of a BlackBox is empty, the real logic happens in firrtl/Emitter.scala
}
