// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.{BaseModule, Param}
import chisel3.internal.BaseBlackBox
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl._
import chisel3.internal.throwException
import chisel3.internal.sourceinfo.{SourceInfo, UnlocatableSourceInfo}
import scala.annotation.nowarn

package internal {

  private[chisel3] abstract class BaseBlackBox extends BaseModule

}

package experimental {

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
  @nowarn("msg=class Port") // delete when Port becomes private
  abstract class ExtModule(val params: Map[String, Param] = Map.empty[String, Param]) extends BaseBlackBox {
    private[chisel3] override def generateComponent(): Option[Component] = {
      require(!_closed, "Can't generate module more than once")
      _closed = true

      val names = nameIds(classOf[ExtModule])

      // Ports are named in the same way as regular Modules
      namePorts(names)

      // All suggestions are in, force names to every node.
      // While BlackBoxes are not supposed to have an implementation, we still need to call
      // _onModuleClose on all nodes (for example, Aggregates use it for recursive naming).
      for (id <- getIds) {
        id._onModuleClose
      }

      closeUnboundIds(names)

      val firrtlPorts = getModulePorts.map { port => Port(port, port.specifiedDirection) }
      val component = DefBlackBox(this, name, firrtlPorts, SpecifiedDirection.Unspecified, params)
      _component = Some(component)
      _component
    }

    private[chisel3] def initializeInParent(parentCompileOptions: CompileOptions): Unit = {
      implicit val sourceInfo = UnlocatableSourceInfo

      for (x <- getModulePorts) {
        pushCommand(DefInvalid(sourceInfo, x.ref))
      }
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
@nowarn("msg=class Port") // delete when Port becomes private
abstract class BlackBox(
  val params: Map[String, Param] = Map.empty[String, Param]
)(
  implicit compileOptions: CompileOptions)
    extends BaseBlackBox {

  // Find a Record port named "io" for purposes of stripping the prefix
  private[chisel3] lazy val _io: Option[Record] =
    this
      .findPort("io")
      .collect { case r: Record => r } // Must be a Record

  // Allow access to bindings from the compatibility package
  protected def _compatIoPortBound() = _io.exists(portsContains(_))

  private[chisel3] override def generateComponent(): Option[Component] = {
    _compatAutoWrapPorts() // pre-IO(...) compatibility hack

    // Restrict IO to just io, clock, and reset
    if (!_io.exists(portsContains)) {
      throwException(s"BlackBox '$this' must have a port named 'io' of type Record wrapped in IO(...)!")
    }

    require(portsSize == 1, "BlackBox must only have one IO, called `io`")

    require(!_closed, "Can't generate module more than once")
    _closed = true

    val io = _io.get

    val namedPorts = io.elements.toSeq.reverse // ListMaps are stored in reverse order

    // There is a risk of user improperly attempting to connect directly with io
    // Long term solution will be to define BlackBox IO differently as part of
    //   it not descending from the (current) Module
    for ((name, port) <- namedPorts) {
      // We are setting a 'fake' ref for io, so that cloneType works but if a user connects to io, it still fails.
      this.findPort("io").get.setRef(ModuleIO(internal.ViewParent, ""), force = true)
      // We have to force override the _ref because it was set during IO binding
      port.setRef(ModuleIO(this, _namespace.name(name)), force = true)
    }

    // We need to call forceName and onModuleClose on all of the sub-elements
    // of the io bundle, but NOT on the io bundle itself.
    // Doing so would cause the wrong names to be assigned, since their parent
    // is now the module itself instead of the io bundle.
    for (id <- getIds; if id ne io) {
      id._onModuleClose
    }

    val firrtlPorts = namedPorts.map { namedPort => Port(namedPort._2, namedPort._2.specifiedDirection) }
    val component = DefBlackBox(this, name, firrtlPorts, io.specifiedDirection, params)
    _component = Some(component)
    _component
  }

  private[chisel3] def initializeInParent(parentCompileOptions: CompileOptions): Unit = {
    for ((_, port) <- _io.map(_.elements).getOrElse(Nil)) {
      pushCommand(DefInvalid(UnlocatableSourceInfo, port.ref))
    }
  }
}
