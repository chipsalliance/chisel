// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.SpecifiedDirection
import chisel3.experimental.{BaseModule, UnlocatableSourceInfo}
import chisel3.internal.BaseBlackBox
import chisel3.internal.firrtl.ir.{Component, DefBlackBox, Port}
import chisel3.layer.Layer

/** Parameters for BlackBoxes */
abstract class Param
case class IntParam(value: BigInt) extends Param
case class DoubleParam(value: Double) extends Param
case class StringParam(value: String) extends Param

/** Creates a parameter from the Printable's resulting format String */
case class PrintableParam(value: chisel3.Printable, context: BaseModule) extends Param

/** Unquoted String */
case class RawParam(value: String) extends Param

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
abstract class ExtModule(
  val params:                               Map[String, Param] = Map.empty[String, Param],
  override protected final val knownLayers: Seq[Layer] = Seq.empty[Layer]
) extends BaseBlackBox {
  private[chisel3] override def generateComponent(): Option[Component] = {
    require(!_closed, "Can't generate module more than once")

    evaluateAtModuleBodyEnd()

    _closed = true

    // Ports are named in the same way as regular Modules
    namePorts()

    val firrtlPorts = getModulePortsAndLocators.map { case (port, _, associations) =>
      Port(port, port.specifiedDirection, associations, UnlocatableSourceInfo)
    }
    val component = DefBlackBox(this, name, firrtlPorts, SpecifiedDirection.Unspecified, params, getKnownLayers)
    _component = Some(component)
    _component
  }

  private[chisel3] def initializeInParent(): Unit = {
    implicit val sourceInfo = UnlocatableSourceInfo
  }
}
