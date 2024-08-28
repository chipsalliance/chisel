// SPDX-License-Identifier: Apache-2.0

package chisel3

import circt.stage.ChiselStage

/** Useful utility methods for generating docs */
package object docs {
  def emitSystemVerilog(gen: => RawModule): String = {
    val prettyArgs = Array("-disable-all-randomization", "-strip-debug-info", "-default-layer-specialization=enable")
    ChiselStage.emitSystemVerilog(gen, firtoolOpts = prettyArgs)
  }
}
