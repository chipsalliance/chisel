// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.{BaseModule, Param}
import chisel3.layer.Layer
import chisel3.internal.BaseBlackBox
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl.ir._
import chisel3.internal.throwException
import chisel3.experimental.{SourceInfo, UnlocatableSourceInfo}

import scala.collection.mutable

package internal {

  private[chisel3] abstract class BaseBlackBox extends BaseModule {
    // Hack to make it possible to run the AddDedupAnnotation
    // pass. Because of naming bugs in imported definitions in D/I, it
    // is not possible to properly name EmptyExtModule created from
    // Defintions. See unit test SeparateElaborationSpec #4.a
    private[chisel3] def _isImportedDefinition: Boolean = false

    /** User-provided information about what layers are known to this `BlackBox`.
      *
      * E.g., if this was a [[BlackBox]] that points at Verilog built from
      * _another_ Chisel elaboration, then this would be the layers that were
      * defined in that circuit.
      *
      * @note This will cause the emitted FIRRTL to include the `knownlayer`
      * keyword on the `extmodule` declaration.
      */
    protected def knownLayers: Seq[Layer]

    // Internal tracking of _knownLayers.  This can be appended to with
    // `addKnownLayer` which happens if you use `addLayer` inside an external
    // module.
    private val _knownLayers: mutable.LinkedHashSet[Layer] = mutable.LinkedHashSet.empty[Layer]

    /** Add a layer to list of knownLayers for this module. */
    private[chisel3] def addKnownLayer(layer: Layer) = {
      var currentLayer: Layer = layer
      while (currentLayer != Layer.Root && !_knownLayers.contains(currentLayer)) {
        val layer = currentLayer
        val parent = layer.parent

        _knownLayers += layer
        currentLayer = parent
      }
    }

    /** Get the known layers.
      *
      * @throw IllegalArgumentException if the module is not closed
      */
    private[chisel3] def getKnownLayers: Seq[Layer] = {
      require(isClosed, "Can't get layers before module is closed")
      _knownLayers.toSeq
    }

    knownLayers.foreach(layer.addLayer)
  }
}

package experimental {

  /** Parameters for BlackBoxes */
  sealed abstract class Param
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
    require(getAssociations.isEmpty, "BlackBoxes cannot support associations at this time, use an ExtModule")
    val firrtlPorts = namedPorts.map { namedPort =>
      Port(namedPort._2, namedPort._2.specifiedDirection, Seq.empty, UnlocatableSourceInfo)
    }

    val component = DefBlackBox(this, name, firrtlPorts, io.specifiedDirection, params, getKnownLayers)
    _component = Some(component)
    _component
  }

  private[chisel3] def initializeInParent(): Unit = {}
}
