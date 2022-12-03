// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._

class SourceLocatorSpec extends ChiselFunSpec with Utils {
  describe("(0) Relative source paths") {
    it("(0.a): are emitted by default relative to `user-dir`") {
      class Top extends Module {
        val w = Wire(UInt(1.W))
      }
      val (chirrtl, _) = getFirrtlAndAnnos(new Top)
      chirrtl.serialize should include("@[src/test/scala/chiselTests/SourceLocatorSpec.scala")
    }
  }
}
