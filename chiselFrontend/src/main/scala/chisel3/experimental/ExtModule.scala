package chisel3.experimental

import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.UnlocatableSourceInfo
import chisel3.{BaseBlackBox, CompileOptions, SpecifiedDirection}

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
  private[chisel3] override def generateComponent(): Component = {
    require(!_closed, "Can't generate module more than once")
    _closed = true

    val names = nameIds(classOf[ExtModule])

    // Name ports based on reflection
    for (port <- getModulePorts) {
      require(names.contains(port), s"Unable to name port $port in $this")
      port.setRef(ModuleIO(this, _namespace.name(names(port))))
    }

    // All suggestions are in, force names to every node.
    // While BlackBoxes are not supposed to have an implementation, we still need to call
    // _onModuleClose on all nodes (for example, Aggregates use it for recursive naming).
    for (id <- getIds) {
      id._onModuleClose
    }

    val firrtlPorts = getModulePorts map {port => Port(port, port.specifiedDirection)}
    val component = DefBlackBox(this, name, firrtlPorts, SpecifiedDirection.Unspecified, params)
    _component = Some(component)
    component
  }

  private[chisel3] def initializeInParent(parentCompileOptions: CompileOptions): Unit = {
    implicit val sourceInfo = UnlocatableSourceInfo

    for (x <- getModulePorts) {
      pushCommand(DefInvalid(sourceInfo, x.ref))
    }
  }
}
