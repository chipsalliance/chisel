// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chiselTests.{ChiselFlatSpec, MatchesAndOmits}
import circt.stage.ChiselStage

import chisel3._

class IntrinsicSpec extends ChiselFlatSpec with MatchesAndOmits {
  behavior.of("Intrinsics")

  it should "support a simple intrinsic statement" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      Intrinsic("test")()
    })

    matchesAndOmits(chirrtl)("intrinsic(test)")()
  }

  it should "support intrinsic statements with arguments" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val f = IO(UInt(3.W))
      val g = IO(UInt(5.W))
      Intrinsic("test")(f, g)
    })

    matchesAndOmits(chirrtl)("intrinsic(test, f, g)")()
  }
  it should "support intrinsic statements with parameters and arguments" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val f = IO(UInt(3.W))
      val g = IO(UInt(5.W))
      Intrinsic("test", "Foo" -> 5)(f, g)
    })

    matchesAndOmits(chirrtl)("intrinsic(test<Foo = 5>, f, g)")()
  }

  it should "support intrinsic expressions" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val f = IO(UInt(3.W))
      val g = IO(UInt(5.W))
      val test = IntrinsicExpr("test", UInt(32.W))(f, g) + 3.U
    })

    matchesAndOmits(chirrtl)(" = intrinsic(test : UInt<32>, f, g)")()
  }

  it should "support intrinsic expressions with parameters and arguments" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val f = IO(UInt(3.W))
      val g = IO(UInt(5.W))
      val test = IntrinsicExpr("test", UInt(32.W), "foo" -> "bar", "x" -> 5)(f, g) + 3.U
    })

    matchesAndOmits(chirrtl)(" = intrinsic(test<foo = \"bar\", x = 5> : UInt<32>, f, g)")()
  }

  it should "preserve parameter order" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val test = IntrinsicExpr("test", UInt(32.W), "x" -> 5, "foo" -> 3)()
    })

    matchesAndOmits(chirrtl)(" = intrinsic(test<x = 5, foo = 3> : UInt<32>)")()
  }

  it should "requite unique parameter names" in {
    intercept[IllegalArgumentException] {
      ChiselStage.emitCHIRRTL(new RawModule {
        val test = IntrinsicExpr("test", UInt(32.W), "foo" -> 5, "foo" -> 3)()
      })
    }.getMessage() should include(
      "parameter names must be unique"
    )
  }
}
