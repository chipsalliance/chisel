// See LICENSE for license details.

package chisel.core

import chisel.internal.Builder.pushCommand
import chisel.internal.firrtl.{ModuleIO, DefInvalid}
import chisel.internal.sourceinfo.SourceInfo

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
  override private[core] def ports = io.elements.toSeq

  // Do not do reflective naming of internal signals, just name io
  override private[core] def setRefs(): this.type = {
    for ((name, port) <- ports) {
      port.setRef(ModuleIO(this, _namespace.name(name)))
    }
    // setRef is not called on the actual io.
    // There is a risk of user improperly attempting to connect directly with io
    // Long term solution will be to define BlackBox IO differently as part of
    //   it not descending from the (current) Module
    this
  }

  // Don't setup clock, reset
  // Cann't invalide io in one bunch, must invalidate each part separately
  override private[core] def setupInParent(implicit sourceInfo: SourceInfo): this.type = _parent match {
    case Some(p) => {
      // Just init instance inputs
      for((_,port) <- ports) pushCommand(DefInvalid(sourceInfo, port.ref))
      this
    }
    case None => this
  }

  // Using null is horrible but these signals SHOULD NEVER be used:
  override val clock = null
  override val reset = null
}
