// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chiselTests.{ChiselFlatSpec, MatchesAndOmits}
import circt.stage.ChiselStage

import chisel3._

import scala.collection.SeqMap

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
      Intrinsic("test", SeqMap("Foo" -> 5))(f, g)
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
      val test = IntrinsicExpr("test", SeqMap("foo" -> "bar", "x" -> 5), UInt(32.W))(f, g) + 3.U
    })

    matchesAndOmits(chirrtl)(" = intrinsic(test<foo = \"bar\", x = 5> : UInt<32>, f, g)")()
  }
}
