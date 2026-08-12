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
        val message = IO(Input(Property[String]()))
        prop.assert("must be true")
        prop.assert(message)
      })
      .fileCheck() {
        """|CHECK: input prop : Bool
           |CHECK: input message : String
           |CHECK: propassert prop, String("must be true")
           |CHECK: propassert prop, message
           |""".stripMargin
      }
  }

  it should "work in a Class" in {
    ChiselStage
      .emitCHIRRTL(new RawModule {
        Definition(new Class {
          override def desiredName = "TestClass"
          val prop = IO(Input(Property[Boolean]()))
          val message = IO(Input(Property[String]()))
          prop.assert("must be true")
          prop.assert(message)
        })
      })
      .fileCheck() {
        """|CHECK: class TestClass :
           |CHECK: input prop : Bool
           |CHECK: input message : String
           |CHECK: propassert prop, String("must be true")
           |CHECK: propassert prop, message
           |""".stripMargin
      }
  }

  it should "compile to SystemVerilog" in {
    class Foo extends RawModule {
      val prop = IO(Input(Property[Boolean]()))
      val username = IO(Input(Property[String]()))
      prop.assert("must be true")
      prop.assert(Property("Hello ") ++ username ++ Property("!"))
    }
    ChiselStage.emitSystemVerilog(new Foo)
  }
}
