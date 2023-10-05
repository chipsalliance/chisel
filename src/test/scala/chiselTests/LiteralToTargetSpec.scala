// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import circt.stage.ChiselStage
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.should.Matchers

class LiteralToTargetSpec extends AnyFreeSpec with Matchers {

  "Literal Data should fail to be converted to ReferenceTarget" in {

    (the[ChiselException] thrownBy {

      class Bar extends RawModule {
        val a = 1.U
      }

      class Foo extends RawModule {
        val bar = Module(new Bar)
        bar.a.toTarget
      }

      ChiselStage.emitCHIRRTL(new Foo)
    } should have).message("Illegal component name: UInt<1>(0h01) (note: literals are illegal)")
  }
}
