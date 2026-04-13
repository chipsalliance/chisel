// SPDX-License-Identifier: Apache-2.0

package chiselTests.properties

import chisel3._
import chisel3.experimental.hierarchy.Definition
import chisel3.properties.{Class, Property}
import chisel3.testing.scalatest.FileCheck
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class PropertyEqualitySpec extends AnyFlatSpec with Matchers with FileCheck {
  behavior.of("PropertyEquality")

  it should "work with Property[Boolean]" in {
    ChiselStage
      .emitCHIRRTL(new RawModule {
        val a = IO(Input(Property[Boolean]()))
        val b = IO(Input(Property[Boolean]()))
        val eq = IO(Output(Property[Boolean]()))
        eq := a === b
      })
      .fileCheck() {
        """|CHECK: input a : Bool
           |CHECK: input b : Bool
           |CHECK: output eq : Bool
           |CHECK: wire {{.+}} : Bool
           |CHECK: propassign {{.+}}, prop_eq(a, b)
           |CHECK: propassign eq, {{.+}}
           |""".stripMargin
      }
  }

  it should "work with Property[Int]" in {
    ChiselStage
      .emitCHIRRTL(new RawModule {
        val a = IO(Input(Property[Int]()))
        val b = IO(Input(Property[Int]()))
        val eq = IO(Output(Property[Boolean]()))
        eq := a === b
      })
      .fileCheck() {
        """|CHECK: input a : Integer
           |CHECK: input b : Integer
           |CHECK: output eq : Bool
           |CHECK: wire {{.+}} : Bool
           |CHECK: propassign {{.+}}, prop_eq(a, b)
           |CHECK: propassign eq, {{.+}}
           |""".stripMargin
      }
  }

  it should "work with Property[String]" in {
    ChiselStage
      .emitCHIRRTL(new RawModule {
        val a = IO(Input(Property[String]()))
        val b = IO(Input(Property[String]()))
        val eq = IO(Output(Property[Boolean]()))
        eq := a === b
      })
      .fileCheck() {
        """|CHECK: input a : String
           |CHECK: input b : String
           |CHECK: output eq : Bool
           |CHECK: wire {{.+}} : Bool
           |CHECK: propassign {{.+}}, prop_eq(a, b)
           |CHECK: propassign eq, {{.+}}
           |""".stripMargin
      }
  }

  it should "compile to SystemVerilog" in {
    ChiselStage.emitSystemVerilog(new RawModule {
      val a = IO(Input(Property[String]()))
      val b = IO(Input(Property[String]()))
      val eq = IO(Output(Property[Boolean]()))
      eq := a === b
    })
  }
}
