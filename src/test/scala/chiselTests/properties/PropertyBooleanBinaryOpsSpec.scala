// SPDX-License-Identifier: Apache-2.0

package chiselTests.properties

import chisel3._
import chisel3.properties.Property
import chisel3.testing.scalatest.FileCheck
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class PropertyBooleanBinaryOpsSpec extends AnyFlatSpec with Matchers with FileCheck {
  behavior.of("PropertyBooleanBinaryOps")

  it should "support AND of two boolean properties" in {
    ChiselStage
      .emitCHIRRTL(new RawModule {
        val a = IO(Input(Property[Boolean]()))
        val b = IO(Input(Property[Boolean]()))
        val out = IO(Output(Property[Boolean]()))
        out := a && b
      })
      .fileCheck() {
        """|CHECK: input a : Bool
           |CHECK: input b : Bool
           |CHECK: output out : Bool
           |CHECK: wire {{.+}} : Bool
           |CHECK: propassign {{.+}}, bool_and(a, b)
           |CHECK: propassign out, {{.+}}
           |""".stripMargin
      }
  }

  it should "support OR of two boolean properties" in {
    ChiselStage
      .emitCHIRRTL(new RawModule {
        val a = IO(Input(Property[Boolean]()))
        val b = IO(Input(Property[Boolean]()))
        val out = IO(Output(Property[Boolean]()))
        out := a || b
      })
      .fileCheck() {
        """|CHECK: input a : Bool
           |CHECK: input b : Bool
           |CHECK: output out : Bool
           |CHECK: wire {{.+}} : Bool
           |CHECK: propassign {{.+}}, bool_or(a, b)
           |CHECK: propassign out, {{.+}}
           |""".stripMargin
      }
  }

  it should "support XOR of two boolean properties" in {
    ChiselStage
      .emitCHIRRTL(new RawModule {
        val a = IO(Input(Property[Boolean]()))
        val b = IO(Input(Property[Boolean]()))
        val out = IO(Output(Property[Boolean]()))
        out := a ^ b
      })
      .fileCheck() {
        """|CHECK: input a : Bool
           |CHECK: input b : Bool
           |CHECK: output out : Bool
           |CHECK: wire {{.+}} : Bool
           |CHECK: propassign {{.+}}, bool_xor(a, b)
           |CHECK: propassign out, {{.+}}
           |""".stripMargin
      }
  }

  it should "support NOT of a boolean property via unary_!" in {
    ChiselStage
      .emitCHIRRTL(new RawModule {
        val a = IO(Input(Property[Boolean]()))
        val out = IO(Output(Property[Boolean]()))
        out := !a
      })
      .fileCheck() {
        """|CHECK: input a : Bool
           |CHECK: output out : Bool
           |CHECK: propassign {{.+}}, bool_xor(a, Bool(true))
           |CHECK: propassign out, {{.+}}
           |""".stripMargin
      }
  }

  it should "support NOT idiom via XOR with literal true" in {
    ChiselStage
      .emitCHIRRTL(new RawModule {
        val a = IO(Input(Property[Boolean]()))
        val out = IO(Output(Property[Boolean]()))
        out := a ^ Property(true)
      })
      .fileCheck() {
        """|CHECK: input a : Bool
           |CHECK: output out : Bool
           |CHECK: propassign {{.+}}, bool_xor(a, Bool(true))
           |CHECK: propassign out, {{.+}}
           |""".stripMargin
      }
  }

  it should "support chaining boolean property operations" in {
    ChiselStage
      .emitCHIRRTL(new RawModule {
        val a = IO(Input(Property[Boolean]()))
        val b = IO(Input(Property[Boolean]()))
        val c = IO(Input(Property[Boolean]()))
        val out = IO(Output(Property[Boolean]()))
        out := (a && b) || c
      })
      .fileCheck() {
        """|CHECK: input a : Bool
           |CHECK: input b : Bool
           |CHECK: input c : Bool
           |CHECK: output out : Bool
           |CHECK: propassign {{.+}}, bool_and(a, b)
           |CHECK: propassign {{.+}}, bool_or({{.+}}, c)
           |CHECK: propassign out, {{.+}}
           |""".stripMargin
      }
  }

  // Just test that this compiles all the way to Verilog.
  it should "compile to SystemVerilog" in {
    ChiselStage.emitSystemVerilog(new RawModule {
      val a = IO(Input(Property[Boolean]()))
      val b = IO(Input(Property[Boolean]()))
      val out = IO(Output(Property[Boolean]()))
      out := (a && b) || (a ^ b)
    })
  }
}
