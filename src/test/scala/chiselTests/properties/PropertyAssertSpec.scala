// SPDX-License-Identifier: Apache-2.0

package chiselTests.properties

import chisel3._
import chisel3.experimental.hierarchy.Definition
import chisel3.properties.{Class, Property}
import chisel3.testing.scalatest.FileCheck
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class PropertyAssertSpec extends AnyFlatSpec with Matchers with FileCheck {
  behavior.of("PropertyAssert")

  it should "work in a RawModule" in {
    ChiselStage
      .emitCHIRRTL(new RawModule {
        val prop = IO(Input(Property[Boolean]()))
        prop.assert("must be true")
      })
      .fileCheck() {
        """|CHECK: input prop : Bool
           |CHECK: propassert prop, "must be true"
           |""".stripMargin
      }
  }

  it should "work in a Class" in {
    ChiselStage
      .emitCHIRRTL(new RawModule {
        Definition(new Class {
          override def desiredName = "TestClass"
          val prop = IO(Input(Property[Boolean]()))
          prop.assert("must be true")
        })
      })
      .fileCheck() {
        """|CHECK: class TestClass :
           |CHECK: input prop : Bool
           |CHECK: propassert prop, "must be true"
           |""".stripMargin
      }
  }

  it should "compile to SystemVerilog" in {
    ChiselStage.emitSystemVerilog(new RawModule {
      val prop = IO(Input(Property[Boolean]()))
      prop.assert("must be true")
    })
  }
}
