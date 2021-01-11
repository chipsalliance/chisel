// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.stage.ChiselGeneratorAnnotation

import org.scalatest.{
  FreeSpec,
  Matchers
}

class LiteralToTargetSpec extends FreeSpec with Matchers {

  "Literal Data should fail to be converted to ReferenceTarget" in {

    the [chisel3.internal.ChiselException] thrownBy {

      class Bar extends RawModule {
        val a = 1.U
      }

      class Foo extends RawModule {
        val bar = Module(new Bar)
        bar.a.toTarget
      }

      ChiselGeneratorAnnotation(() => new Foo).elaborate
    } should have message "Illegal component name: UInt<1>(\"h01\") (note: literals are illegal)"
  }
}
