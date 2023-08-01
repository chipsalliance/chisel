// SPDX-License-Identifier: Apache-2.0

package chiselTests

import circt.stage.ChiselStage.emitCHIRRTL
import chisel3._
import chisel3.experimental.SourceLine
import firrtl.ir.FileInfo

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

  describe("(1) Source locators with special characters") {
    val filename = "I need escaping\n\\\t].scala"
    val escaped = "I need escaping\\n\\\\\\t\\].scala"
    val info = SourceLine(filename, 123, 456)
    it("(1.a): are properly escaped when converting to FIRRTL") {
      val firrtl = FileInfo.fromUnescaped(filename)
      firrtl should equal(FileInfo(escaped))
    }
    it("(1.b): are properly escaped to FIRRTL through Chisel elaboration") {
      implicit val sl = info

      val chirrtl = emitCHIRRTL(new RawModule {
        val in = IO(Input(UInt(8.W)))
        val out = IO(Output(UInt(8.W)))
        out := in
      })
      chirrtl should include(escaped)
    }
    it("(1.c): can be properly unescaped") {
      val escapedInfo = FileInfo(escaped)
      escapedInfo.unescaped should equal(filename)
    }
  }
}
