// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.{SourceInfo, UnlocatableSourceInfo}
import chisel3.internal.{Builder, HasId}
import chisel3.internal.firrtl.ir.{LayerBlockBegin, LayerBlockEnd, Node}
import chisel3.util.simpleClassName
import scala.annotation.tailrec
import scala.collection.mutable.LinkedHashSet

/** This object contains Chisel language features for creating layers.  Layers
  * are collections of hardware that are not always present in the circuit.
  * Layers are intended to be used to hold verification or debug code.
  */
object layer {

  /** Enumerations of different optional layer conventions.  A layer convention
    * says how a given layer should be lowered to Verilog.
    */
  object Convention {
    sealed trait Type

    /** Internal type used as the parent of all layers. */
    private[chisel3] case object Root extends Type

    /** The layer should be lowered to a SystemVerilog `bind`. */
    case object Bind extends Type
  }

  /** A layer declaration.
    *
    * @param convention how this layer should be lowered
    * @param _parent the parent layer, if any
    */
  abstract class Layer(val convention: Convention.Type)(implicit _parent: Layer, _sourceInfo: SourceInfo) {
    self: Singleton =>

    /** This establishes a new implicit val for any nested layers. */
    protected final implicit val thiz: Layer = this

    private[chisel3] def parent: Layer = _parent

    private[chisel3] def sourceInfo: SourceInfo = _sourceInfo

    private[chisel3] def name: String = simpleClassName(this.getClass())

    private[chisel3] val fullName: String = parent match {
      case null       => "<root>"
      case Layer.Root => name
      case _          => s"${parent.fullName}.$name"
    }

    @tailrec
    final private[chisel3] def canWriteTo(that: Layer): Boolean = that match {
      case null              => false
      case _ if this == that => true
      case _                 => this.canWriteTo(that.parent)
    }
  }

  object Layer {
    private[chisel3] case object Root extends Layer(Convention.Root)(null, UnlocatableSourceInfo)
    implicit val root: Layer = Root
  }

  /** Add a layer and all of its parents to the Builder.  This lets the Builder
    * know that this layer was used and should be emitted in the FIRRTL.
    */
  private[chisel3] def addLayer(layer: Layer) = {
    var currentLayer: Layer = layer
    while (currentLayer != Layer.Root && !Builder.layers.contains(currentLayer)) {
      val layer = currentLayer
      val parent = layer.parent

      Builder.layers += layer
      currentLayer = parent
    }
  }

  /** Create a new layer block.  This is hardware that will be enabled
    * collectively when the layer is enabled.
    *
    * @param layer the layer this block is associated with
    * @param thunk the Chisel code that goes into the layer block
    * @param sourceInfo a source locator
    */
  def block[A](
    layer: Layer
  )(thunk: => A
  )(
    implicit sourceInfo: SourceInfo
  ): Unit = {
    Builder.pushCommand(LayerBlockBegin(sourceInfo, layer))
    addLayer(layer)
    require(
      Builder.layerStack.head == layer.parent,
      s"nested layer '${layer.name}' must be wrapped in parent layer '${layer.parent.name}'"
    )
    Builder.layerStack = layer :: Builder.layerStack
    thunk
    Builder.pushCommand(LayerBlockEnd(sourceInfo))
    Builder.layerStack = Builder.layerStack.tail
  }

  /** Call this function from within a `Module` body to enable this layer globally for that module. */
  final def enable(layer: Layer): Unit = layer match {
    case Layer.Root =>
    case _ =>
      addLayer(layer)
      Builder.enabledLayers += layer
  }

}
