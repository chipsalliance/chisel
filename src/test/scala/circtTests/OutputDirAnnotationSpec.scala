// SPDX-License-Identifier: Apache-2.0

package circtTests

import chisel3._
import circt.outputDir
import circt.stage.ChiselStage
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers
// import chiselTests.experimental.hierarchy.Utils

class OutputDirAnnotationSpec extends AnyFunSpec with Matchers {
  describe("output directory annotation works") {
    it("should put module where requested") {
      class TestModule extends RawModule with Public {}
      class ParentModule extends RawModule {
        val test = outputDir(Module(new TestModule), "foo")
      }

      val chirrtl = ChiselStage.emitCHIRRTL(new ParentModule)
      (chirrtl.split('\n').map(_.takeWhile(_ != '@').trim) should contain).allOf(
        """"class":"circt.OutputDirAnnotation",""",
        """"target":"~|TestModule",""",
        """"dirname":"foo""""
      )

      val sv = ChiselStage.emitSystemVerilog(new ParentModule)
      sv should include(""""foo/TestModule.sv"""")
    }
  }
}
