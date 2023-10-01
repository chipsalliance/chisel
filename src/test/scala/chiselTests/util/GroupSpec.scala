// SPDX-License-Identifier: Apache-2.0

package chiselTests.util

import chisel3._
import chiselTests.{ChiselFlatSpec, MatchesAndOmits, Utils}
import _root_.circt.stage.ChiselStage

class GroupSpec extends ChiselFlatSpec with Utils with MatchesAndOmits {

  "Groups" should "allow for creation of a group and nested groups" in {

    object A extends group.Declaration(group.Convention.Bind) {
      object B extends group.Declaration(group.Convention.Bind)
    }

    class Foo extends RawModule {
      val a = IO(Input(Bool()))

      group(A) {
        val w = WireInit(a)
        dontTouch(w)
        group(A.B) {
          val x = WireInit(w)
          dontTouch(x)
        }
      }

      val y = Wire(Bool())
      y := DontCare
    }

    val chirrtl = ChiselStage.emitCHIRRTL(new Foo, Array("--full-stacktrace"))

    info("CHIRRTL emission looks coorect")
    // TODO: Switch to FileCheck for this testing.  This is going to miss all sorts of ordering issues.
    matchesAndOmits(chirrtl)(
      "declgroup A, bind :",
      "declgroup B, bind :",
      "group A :",
      "wire w : UInt<1>",
      "group B :",
      "wire x : UInt<1>",
      "wire y : UInt<1>"
    )()
  }

  "Groups error checking" should "require that a nested group definition matches its declaration nesting" in {

    object A extends group.Declaration(group.Convention.Bind) {
      object B extends group.Declaration(group.Convention.Bind)
    }

    class Foo extends RawModule {
      group(A.B) {
        val a = Wire(Bool())
      }
    }

    intercept[IllegalArgumentException] { ChiselStage.emitCHIRRTL(new Foo) }.getMessage() should include(
      "nested group 'B' must be wrapped in parent group 'A'"
    )

  }

}
