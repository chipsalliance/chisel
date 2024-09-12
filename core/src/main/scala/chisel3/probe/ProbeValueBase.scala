// SPDX-License-Identifier: Apache-2.0

package chisel3.probe

import chisel3.Data
import chisel3.internal.Builder
import chisel3.internal.binding.OpBinding
import chisel3.internal.firrtl.ir.{ProbeExpr, RWProbeExpr}
import chisel3.experimental.{requireIsHardware, SourceInfo}

private[chisel3] trait ProbeValueBase {
  protected def apply[T <: Data](source: T, writable: Boolean)(implicit sourceInfo: SourceInfo): T = {
    requireIsHardware(source)
    // construct probe to return with cloned info
    val clone = if (writable) RWProbe(source.cloneType) else Probe(source.cloneType)
    clone.bind(OpBinding(Builder.forcedUserModule, Builder.currentWhen))
    if (writable) {
      if (source.isLit) {
        Builder.error("Cannot get a probe value from a literal.")
      }
      clone.setRef(RWProbeExpr(source.ref))
    } else {
      val ref = if (source.isLit) {
        val intermed = chisel3.Wire(source.cloneType)
        intermed := source
        intermed.suggestName("lit_probe_val")
        intermed.ref
      } else { source.ref }
      clone.setRef(ProbeExpr(ref))
    }
    clone
  }
}
