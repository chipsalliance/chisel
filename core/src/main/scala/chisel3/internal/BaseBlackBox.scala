// SPDX-License-Identifier: Apache-2.0

package chisel3.internal

import chisel3.experimental.BaseModule
import chisel3.layer.{addLayer, Layer}
import scala.collection.mutable

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

  knownLayers.foreach(addLayer)
}
