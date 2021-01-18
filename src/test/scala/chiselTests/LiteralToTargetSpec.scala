// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.stage.ChiselStage

import org.scalatest._
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.should.Matchers


class LiteralToTargetSpec extends AnyFreeSpec with Matchers {

  "Literal Data should fail to be converted to ReferenceTarget" in {

    the [chisel3.internal.ChiselException] thrownBy {

      class Bar extends RawModule {
        val a = 1.U
      }

      class Foo extends RawModule {
        val bar = Module(new Bar)
        bar.a.toTarget
      }

      ChiselStage.elaborate(new Foo)
    } should have message "Illegal component name: UInt<1>(\"h01\") (note: literals are illegal)"
  }
}
