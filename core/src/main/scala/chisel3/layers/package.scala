package chisel3

/** This package contains common [[layer.Layer]]s used by Chisel generators. */
package object layers {

  /** This is a list of layers that will _always_ be included in the design. This
    * is done to provide predictability of what layers will be availble.
    *
    * This list is not user-extensible. If a user wants to have similar behavior
    * for their design, then they should use [[chisel3.layer.addLayer]] API to
    * add the layers that they always want to see in their output.
    */
  val defaultLayers: Seq[layer.Layer] = Seq(
    Verification,
    Verification.Assert,
    Verification.Assert.Temporal,
    Verification.Assume,
    Verification.Assume.Temporal,
    Verification.Cover,
    Verification.Cover.Temporal
  )

}
