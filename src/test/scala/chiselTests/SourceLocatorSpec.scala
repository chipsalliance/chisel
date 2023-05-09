// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._

class SourceLocatorSpec extends ChiselFunSpec with Utils {
  describe("(0) Relative source paths") {
    it("(0.a): are emitted by default relative to `user-dir`") {
      class Top extends Module {
        val w = WireInit(UInt(1.W), 0.U)
      }
      val chirrtl = circt.stage.ChiselStage.emitCHIRRTL(new Top)
      chirrtl should include("@[src/test/scala/chiselTests/SourceLocatorSpec.scala")
    }
  }
}
