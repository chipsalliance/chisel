// See LICENSE for license details.

package chisel3.core

import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl.{ModuleIO, DefInvalid}
import chisel3.internal.sourceinfo.SourceInfo
// TODO: remove this once we have CompileOptions threaded through the macro system.
import chisel3.core.ExplicitCompileOptions.NotStrict

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
    // setRef is not called on the actual io.
    // There is a risk of user improperly attempting to connect directly with io
    // Long term solution will be to define BlackBox IO differently as part of
    //   it not descending from the (current) Module
    for ((name, port) <- ports) {
      port.setRef(ModuleIO(this, _namespace.name(name)))
    }
    // We need to call forceName and onModuleClose on all of the sub-elements
    // of the io bundle, but NOT on the io bundle itself.
    // Doing so would cause the wrong names to be assigned, since their parent
    // is now the module itself instead of the io bundle.
    for (id <- _ids; if id ne io) {
      id.forceName(default="_T", _namespace)
      id._onModuleClose
    }
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
