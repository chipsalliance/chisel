// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.BaseModule
import chisel3.layer.Layer
import chisel3.internal.BaseBlackBox
import chisel3.internal.firrtl.ir.{Component, DefBlackBox, ModuleIO, Port}
import chisel3.internal.throwException
import chisel3.experimental.UnlocatableSourceInfo

import scala.collection.mutable

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
@deprecated(
  "use `chisel3.ExtModule` instead. To migrate replace `val io = { ... }` with `val io = FlatIO { ... }`, `object io { ... }; locally { io }`, or manually flatten your IO as `ExtModule` allows calling `IO` more than once.",
  "7.5.0"
)
abstract class BlackBox(
  val params:                               Map[String, Param] = Map.empty[String, Param],
  override protected final val knownLayers: Seq[Layer] = Seq.empty[Layer]
) extends BaseBlackBox {

  // Find a Record port named "io" for purposes of stripping the prefix
  private[chisel3] lazy val _io: Option[Record] =
    this
      .findPort("io")
      .collect { case r: Record => r } // Must be a Record

  // Allow access to bindings from the compatibility package
  protected def _compatIoPortBound() = _io.exists(portsContains(_))

  override def requirements = Seq.empty[String]

  private[chisel3] override def generateComponent(): Option[Component] = {
    // Restrict IO to just io, clock, and reset
    if (!_io.exists(portsContains)) {
      throwException(s"BlackBox '$this' must have a port named 'io' of type Record wrapped in IO(...)!")
    }

    require(portsSize == 1, "BlackBox must only have one IO, called `io`")

    require(!_closed, "Can't generate module more than once")

    evaluateAtModuleBodyEnd()

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

    // Note: BlackBoxes, because they have a single `io` cannot currently
    // support associations because associations require having multiple ports.
    // If this restriction is lifted, then this code should be updated.
    require(!hasAsssociations, "BlackBoxes cannot support associations at this time, use an ExtModule")
    // Get the source info for the io bundle to use for all flattened ports.
    val ioSourceInfo = getModulePortsAndLocators.collectFirst {
      case (port, sourceInfo, _) if port == io => sourceInfo
    }.getOrElse(UnlocatableSourceInfo)
    val firrtlPorts = namedPorts.map { namedPort =>
      Port(namedPort._2, namedPort._2.specifiedDirection, Seq.empty, ioSourceInfo)
    }

    val component = DefBlackBox(
      this,
      name,
      firrtlPorts,
      io.specifiedDirection,
      params,
      getKnownLayers,
      requirements = requirements
    )
    _component = Some(component)
    _component
  }

  private[chisel3] def initializeInParent(): Unit = {}
}
