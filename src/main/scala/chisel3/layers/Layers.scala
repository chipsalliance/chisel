package chisel3.layers

import chisel3.layer.{Convention, Layer}

/** The root [[chisel3.layer.Layer]] for all shared verification collateral. */
object Verification extends Layer(Convention.Bind) {

  /** The [[chisel3.layer.Layer]] where all assertions will be placed. */
  object Assert extends Layer(Convention.Bind)

  /** The [[chisel3.layer.Layer]] where all assumptions will be placed. */
  object Assume extends Layer(Convention.Bind)

  /** The [[chisel3.layer.Layer]] where all covers will be placed. */
  object Cover extends Layer(Convention.Bind)
}
