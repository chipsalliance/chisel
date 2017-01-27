// See LICENSE for license details.

package chisel3.core

import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl._
import chisel3.internal.throwException
import chisel3.internal.sourceinfo.{SourceInfo, UnlocatableSourceInfo}
// TODO: remove this once we have CompileOptions threaded through the macro system.
import chisel3.core.ExplicitCompileOptions.NotStrict

/** Parameters for BlackBoxes */
sealed abstract class Param
case class IntParam(value: BigInt) extends Param
case class DoubleParam(value: Double) extends Param
case class StringParam(value: String) extends Param
/** Unquoted String */
case class RawParam(value: String) extends Param

abstract class BaseBlackBox extends BaseModule

/** Defines a black box, which is a module that can be referenced from within
  * Chisel, but is not defined in the emitted Verilog. Useful for connecting
  * to RTL modules defined outside Chisel.
  *
  * A variant of BlackBox, this has a more consistent naming scheme in allowing
  * multiple top-level IO and does not drop the top prefix.
  *
  * @example
  * Some design require a differential input clock to clock the all design.
  * With the xilinx FPGA for example, a Verilog template named IBUFDS must be
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
  * To instantiate it, a BlackBox can be used like following:
  * {{{
  * import chisel3._
  * import chisel3.experimental._
  *
  * // Example with Xilinx differential buffer IBUFDS
  * class IBUFDS extends ExtModule(Map("DIFF_TERM" -> "TRUE", // Verilog parameters
  *                                    "IOSTANDARD" -> "DEFAULT"
  *                      )) {
  *   val O = IO(Output(Clock()))
  *   val I = IO(Input(Clock()))
  *   val IB = IO(Input(Clock()))
  * }
  * }}}
  * @note The parameters API is experimental and may change
  */
abstract class ExtModule(val params: Map[String, Param] = Map.empty[String, Param]) extends BaseBlackBox {
  private[core] override def generateComponent(): Component = {
    require(!_closed, "Can't generate module more than once")
    _closed = true

    val names = nameIds(classOf[ExtModule])

    // Name ports based on reflection
    for (port <- _ports) {
      require(names.contains(port), s"Unable to name port $port in $this")
      port.setRef(ModuleIO(this, _namespace.name(names(port))))
    }

    // All suggestions are in, force names to every node.
    for (id <- _ids) {
      id.forceName(default="_T", _namespace)
      id._onModuleClose
    }

    val firrtlPorts = for (port <- _ports) yield {
      // Port definitions need to know input or output at top-level.
      // By FIRRTL semantics, 'flipped' becomes an Input
      val direction = if(Data.isFirrtlFlipped(port)) Direction.Input else Direction.Output
      Port(port, direction)
    }

    val component = DefBlackBox(this, name, firrtlPorts, params)
    _component = Some(component)
    component
  }

  private[core] def initializeInParent() {
    implicit val sourceInfo = UnlocatableSourceInfo

    _ports.foreach { x: Data =>
      pushCommand(DefInvalid(sourceInfo, x.ref))
    }
  }
}

/** Defines a black box, which is a module that can be referenced from within
  * Chisel, but is not defined in the emitted Verilog. Useful for connecting
  * to RTL modules defined outside Chisel.
  *
  * @example
  * Some design require a differential input clock to clock the all design.
  * With the xilinx FPGA for example, a Verilog template named IBUFDS must be
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
  * To instantiate it, a BlackBox can be used like following:
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
abstract class BlackBox(val params: Map[String, Param] = Map.empty[String, Param]) extends BaseBlackBox {
  def io: Record

  private[core] override def generateComponent(): Component = {
    _autoWrapPorts()  // pre-IO(...) compatibility hack

    // Restrict IO to just io, clock, and reset
    require(io != null, "BlackBox must have io")
    require(_ports contains io, "BlackBox must have io wrapped in IO(...)")
    require(_ports.size == 1, "BlackBox must only have io as IO")

    require(!_closed, "Can't generate module more than once")
    _closed = true

    val namedPorts = io.elements.toSeq
    // setRef is not called on the actual io.
    // There is a risk of user improperly attempting to connect directly with io
    // Long term solution will be to define BlackBox IO differently as part of
    //   it not descending from the (current) Module
    for ((name, port) <- namedPorts) {
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

    val firrtlPorts = for ((_, port) <- namedPorts) yield {
      // Port definitions need to know input or output at top-level.
      // By FIRRTL semantics, 'flipped' becomes an Input
      val direction = if(Data.isFirrtlFlipped(port)) Direction.Input else Direction.Output
      Port(port, direction)
    }

    val component = DefBlackBox(this, name, firrtlPorts, params)
    _component = Some(component)
    component
  }

  private[core] def initializeInParent() {
    implicit val sourceInfo = UnlocatableSourceInfo

    val namedPorts = io.elements.toSeq
    for ((_, port) <- namedPorts) {
      pushCommand(DefInvalid(sourceInfo, port.ref))
    }
  }
}
