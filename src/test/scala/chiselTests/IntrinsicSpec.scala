// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.experimental.{fromIntToIntParam, fromStringToStringParam}
import circt.stage.ChiselStage
import chisel3.experimental.{fromIntToIntParam, fromStringToStringParam}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class IntrinsicSpec extends AnyFlatSpec with Matchers {
  behavior.of("Intrinsics")

  it should "support a simple intrinsic statement" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      Intrinsic("test")()
    }) should include("intrinsic(test)")
  }

  it should "support intrinsic statements with arguments" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val f = IO(UInt(3.W))
      val g = IO(UInt(5.W))
      Intrinsic("test")(f, g)
    }) should include("intrinsic(test, f, g)")
  }
  it should "support intrinsic statements with parameters and arguments" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val f = IO(UInt(3.W))
      val g = IO(UInt(5.W))
      Intrinsic("test", "Foo" -> 5)(f, g)
    }) should include("intrinsic(test<Foo = 5>, f, g)")
  }

  it should "support intrinsic expressions" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val f = IO(UInt(3.W))
      val g = IO(UInt(5.W))
      val test = IntrinsicExpr("test", UInt(32.W))(f, g) + 3.U
    }) should include(" = intrinsic(test : UInt<32>, f, g)")
  }

  it should "support intrinsic expressions with parameters and arguments" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val f = IO(UInt(3.W))
      val g = IO(UInt(5.W))
      val test = IntrinsicExpr("test", UInt(32.W), "foo" -> "bar", "x" -> 5)(f, g) + 3.U
    }) should include(" = intrinsic(test<foo = \"bar\", x = 5> : UInt<32>, f, g)")
  }

  it should "preserve parameter order" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val test = IntrinsicExpr("test", UInt(32.W), "x" -> 5, "foo" -> 3)()
    }) should include(" = intrinsic(test<x = 5, foo = 3> : UInt<32>)")
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
