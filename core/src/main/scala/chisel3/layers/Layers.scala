package chisel3.layers

import chisel3.experimental.UnlocatableSourceInfo
import chisel3.layer.{Layer, LayerConfig}

/** The root [[chisel3.layer.Layer]] for all shared verification collateral. */
object Verification extends Layer(LayerConfig.Extract())(_parent = Layer.Root, _sourceInfo = UnlocatableSourceInfo) {

  /** The [[chisel3.layer.Layer]] where all assertions will be placed. */
  object Assert extends Layer(LayerConfig.Extract())(_parent = Verification, _sourceInfo = UnlocatableSourceInfo)

  /** The [[chisel3.layer.Layer]] where all assumptions will be placed. */
  object Assume extends Layer(LayerConfig.Extract())(_parent = Verification, _sourceInfo = UnlocatableSourceInfo)

  /** The [[chisel3.layer.Layer]] where all covers will be placed. */
  object Cover extends Layer(LayerConfig.Extract())(_parent = Verification, _sourceInfo = UnlocatableSourceInfo)
}
