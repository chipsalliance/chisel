// SPDX-License-Identifier: Apache-2.0

package chisel3.simulator

/** This struct describes settings related to controlling a Chisel simulation.  Thes
  *
  * These setings are only intended to be associated with Chisel, FIRRTL, and
  * FIRRTL's Verilog ABI and not to do with lower-level control of the FIRRTL
  * compilation itself or the Verilog compilation and simulation.
  *
  * @param layerControl determines which [[chisel3.layer.Layer]]s should be
  * enabled _during Verilog elaboration_.
  */
final class ChiselSettings(
  /** Layers to turn on/off during Verilog elaboration */
  val verilogLayers: LayerControl.Type
)

object ChiselSettings {

  final def default: ChiselSettings = new ChiselSettings(
    verilogLayers = LayerControl.EnableAll
  )

}
