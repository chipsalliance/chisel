// See LICENSE for license details.

package Chisel

import internal.Builder.pushCommand
import internal.firrtl.{ModuleIO, DefInvalid}

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
abstract class BlackBox extends Module {
  // Don't bother taking override_clock|reset, clock/reset locked out anyway
  // TODO: actually implement this.
  def setVerilogParameters(s: String): Unit = {}

  // The body of a BlackBox is empty, the real logic happens in firrtl/Emitter.scala
  // Bypass standard clock, reset, io port declaration by flattening io
  // TODO(twigg): ? Really, overrides are bad, should extend BaseModule....
  override private[Chisel] def ports = io.elements.toSeq

  // Do not do reflective naming of internal signals, just name io
  override private[Chisel] def setRefs(): this.type = {
    for ((name, port) <- ports) {
      port.setRef(ModuleIO(this, _namespace.name(name)))
    }
    io.setRef("") // don't io parts prepended with io_
    this
  }

  // Don't setup clock, reset
  // Cann't invalide io in one bunch, must invalidate each part separately
  override private[Chisel] def setupInParent(): this.type = _parent match {
    case Some(p) => {
      // Just init instance inputs
      for((_,port) <- ports) pushCommand(DefInvalid(port.ref))
      this
    }
    case None => this
  }

  // Using null is horrible but these signals SHOULD NEVER be used:
  override val clock = null
  override val reset = null
}
