package chisel3.layers

import chisel3.experimental.UnlocatableSourceInfo
import chisel3.layer.{CustomOutputDir, Layer, LayerConfig}
import java.nio.file.Paths

/** Trait that adds a `Temporal` layer inside another layer.
  *
  * This temporal layer can used to guard statements which are unsupported,
  * expensive, or otherwise needed to be excluded from normal design
  * verification code in certain tools or environments.  E.g., this is intended
  * to work around lack of support for certain SystemVerilog Assertions in
  * simulators.
  *
  * @note While this is used to provide temporal sub-layers in Chisel's default
  * layers, it is entirely reasonable for users to mix-in this trait into their
  * own user-defined layers to provide similar, recognizable functionality.
  */
trait HasTemporalInlineLayer { this: Layer =>

  /** The [chisel3.layer.Layer]] where complicated assertions that may not be
    * supported by all tools are placed.
      */
  object Temporal
      extends Layer(LayerConfig.Inline)(
        _parent = implicitly[Layer],
        _sourceInfo = UnlocatableSourceInfo
      )
}

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
      with HasTemporalInlineLayer

  /** The [[chisel3.layer.Layer]] where all assumptions will be placed. */
  object Assume
      extends Layer(LayerConfig.Extract(CustomOutputDir(Paths.get("verification", "assume"))))(
        _parent = implicitly[Layer],
        _sourceInfo = UnlocatableSourceInfo
      )
      with HasTemporalInlineLayer

  /** The [[chisel3.layer.Layer]] where all covers will be placed. */
  object Cover
      extends Layer(LayerConfig.Extract(CustomOutputDir(Paths.get("verification", "cover"))))(
        _parent = implicitly[Layer],
        _sourceInfo = UnlocatableSourceInfo
      )
      with HasTemporalInlineLayer
}
