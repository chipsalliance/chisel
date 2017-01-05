// See LICENSE for license details.

package chisel3.core

import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl._
import chisel3.internal.throwException
import chisel3.internal.sourceinfo.SourceInfo
// TODO: remove this once we have CompileOptions threaded through the macro system.
import chisel3.core.ExplicitCompileOptions.NotStrict

/** Parameters for BlackBoxes */
sealed abstract class Param
case class IntParam(value: BigInt) extends Param
case class DoubleParam(value: Double) extends Param
case class StringParam(value: String) extends Param
/** Unquoted String */
case class RawParam(value: String) extends Param

/** Defines a black box, which is a module that can be referenced from within
  * Chisel, but is not defined in the emitted Verilog. Useful for connecting
  * to RTL modules defined outside Chisel.
  *
  * @example
  * Some design require a differential input clock to clock the all design.
  * With xilinx FPGA for example, a verilog template named IBUFDS must be
  * integrated to use differential input:
  * {{{
  *  IBUFDS #(.DIFF_TERM("TRUE"),
  *           .IOSTANDARD("DEFAULT")) ibufds (
  *   .IB(ibufds_IB),
  *   .I(ibufds_I),
  *   .O(ibufds_O)
  *  );
  * }}}
  *
  * To instanciate it, a BlackBox can be used like following:
  * {{{
  * import chisel3._
  * import chisel3.experimental._
  *
  * // Example with Xilinx differential buffer IBUFDS
  * class IBUFDS extends BlackBox(Map("DIFF_TERM" -> "TRUE", // Verilog parameters
  *                                   "IOSTANDARD" -> "DEFAULT"
  *                      )) {
  *   val io = IO(new Bundle {
  *     val O = Output(Clock()) // IO names will be the same
  *     val I = Input(Clock())  // (without 'io_' in prefix)
  *     val IB = Input(Clock()) //
  *   })
  * }
  * }}}
  * @note The parameters API is experimental and may change
  */
abstract class BlackBox(val params: Map[String, Param] = Map.empty[String, Param]) extends Module {

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
