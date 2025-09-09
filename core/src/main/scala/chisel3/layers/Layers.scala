package chisel3.layers

import chisel3.experimental.UnlocatableSourceInfo
import chisel3.layer.{CustomOutputDir, Layer, LayerConfig}
import java.nio.file.Paths

/** The root [[chisel3.layer.Layer]] for all shared verification collateral. */
object Verification
    extends Layer(LayerConfig.Extract(CustomOutputDir(Paths.get("verification"))))(
      _parent = implicitly[Layer],
      _sourceInfo = UnlocatableSourceInfo
    ) {

  /** The [[chisel3.layer.Layer]] where all assertions will be placed. */
  object Assert
      extends Layer(LayerConfig.Extract(CustomOutputDir(Paths.get("verification", "assert"))))(
        _parent = implicitly[Layer],
        _sourceInfo = UnlocatableSourceInfo
      )

  /** The [[chisel3.layer.Layer]] where all assumptions will be placed. */
  object Assume
      extends Layer(LayerConfig.Extract(CustomOutputDir(Paths.get("verification", "assume"))))(
        _parent = implicitly[Layer],
        _sourceInfo = UnlocatableSourceInfo
      )

  /** The [[chisel3.layer.Layer]] where all covers will be placed. */
  object Cover
      extends Layer(LayerConfig.Extract(CustomOutputDir(Paths.get("verification", "cover"))))(
        _parent = implicitly[Layer],
        _sourceInfo = UnlocatableSourceInfo
      )
}
