// SPDX-License-Identifier: Apache-2.0

package chisel3.probe

import chisel3._
import chisel3.Data.ProbeInfo
import chisel3.experimental.{requireIsChiselType, SourceInfo}
import chisel3.internal.{containsProbe, requireNoProbeTypeModifier, Builder}

/** Utilities for creating and working with Chisel types that have a probe or
  * writable probe modifier.
  */
private[chisel3] trait ProbeBase {

  protected def apply[T <: Data](
    source:   => T,
    writable: Boolean,
    _color:   Option[layer.Layer]
  )(
    implicit sourceInfo: SourceInfo
  ): T = {
    val prevId = Builder.idGen.value
    // call Output() to coerce passivity
    val data = Output(source) // should only evaluate source once
    requireNoProbeTypeModifier(data, "Cannot probe a probe.")
    if (containsProbe(data)) {
      Builder.error("Cannot create a probe of an aggregate containing a probe.")
    }
    if (writable && data.isConst) {
      Builder.error("Cannot create a writable probe of a const type.")
    }
    // TODO error if trying to probe a non-passive type
    // https://github.com/chipsalliance/chisel/issues/3609

    val ret: T = if (!data.mustClone(prevId)) data else data.cloneType.asInstanceOf[T]
    // Remap the color if the user is using the ChiselStage --remap-layer option.
    val color = _color.map(c =>
      Builder.inContext match {
        case false => c
        case true  => Builder.layerMap.getOrElse(c, c)
      }
    )
    // Record the layer in the builder if we are in a builder context.
    if (Builder.inContext && color.isDefined) {
      layer.addLayer(color.get)
    }
    setProbeModifier(ret, Some(ProbeInfo(writable, color)))
    ret
  }

  protected def apply[T <: Data](
    source:   => T,
    writable: Boolean
  )(
    implicit sourceInfo: SourceInfo
  ): T = {
    apply(source, writable, None)
  }
}
