// SPDX-License-Identifier: Apache-2.0

package chiselTests.properties

import chisel3._
import chisel3.properties.Property
import chisel3.testing.scalatest.FileCheck
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class PropertyListContainsSpec extends AnyFlatSpec with Matchers with FileCheck {
  behavior.of("PropertyListContains")

  it should "support Property[Seq[String]]" in {
    ChiselStage
      .emitCHIRRTL(new RawModule {
        val names = IO(Input(Property[Seq[String]]()))
        val target = IO(Input(Property[String]()))
        val found = IO(Output(Property[Boolean]()))
        found := names.contains(target)
      })
      .fileCheck() {
        """|CHECK: input names : List<String>
           |CHECK: input target : String
           |CHECK: output found : Bool
           |CHECK: wire {{.+}} : Bool
           |CHECK: propassign {{.+}}, list_contains(names, target)
           |CHECK: propassign found, {{.+}}
           |""".stripMargin
      }
  }

  it should "support Property[List[Int]]" in {
    ChiselStage
      .emitCHIRRTL(new RawModule {
        val numbers = IO(Input(Property[List[Int]]()))
        val target = IO(Input(Property[Int]()))
        val found = IO(Output(Property[Boolean]()))
        found := numbers.contains(target)
      })
      .fileCheck() {
        """|CHECK: input numbers : List<Integer>
           |CHECK: input target : Integer
           |CHECK: output found : Bool
           |CHECK: propassign {{.+}}, list_contains(numbers, target)
           |""".stripMargin
      }
  }

  it should "support Property[Vector[Boolean]]" in {
    ChiselStage
      .emitCHIRRTL(new RawModule {
        val values = IO(Input(Property[Vector[Boolean]]()))
        val target = IO(Input(Property[Boolean]()))
        val found = IO(Output(Property[Boolean]()))
        found := values.contains(target)
      })
      .fileCheck() {
        """|CHECK: input values : List<Bool>
           |CHECK: input target : Bool
           |CHECK: output found : Bool
           |CHECK: propassign {{.+}}, list_contains(values, target)
           |""".stripMargin
      }
  }

  it should "support a literal list and compose with boolean property operations" in {
    ChiselStage
      .emitCHIRRTL(new RawModule {
        val target = IO(Input(Property[String]()))
        val enable = IO(Input(Property[Boolean]()))
        val found = IO(Output(Property[Boolean]()))
        found := Property[Seq[String]](Seq("cold", "warm")).contains(target) && enable
      })
      .fileCheck() {
        """|CHECK: input target : String
           |CHECK: input enable : Bool
           |CHECK: output found : Bool
           |CHECK: propassign {{.+}}, list_contains({{.+}}, target)
           |CHECK: propassign {{.+}}, bool_and({{.+}}, enable)
           |CHECK: propassign found, {{.+}}
           |""".stripMargin
      }
  }

  it should "reject mismatched or unsupported element types and non-sequence properties" in {
    assertTypeError("""
      import chisel3._
      import chisel3.properties.Property
      Property[Seq[String]]().contains(Property[Int]())
    """)
    assertTypeError("""
      import chisel3._
      import chisel3.properties.Property
      Property[Int]().contains(Property[Int]())
    """)
    assertTypeError("""
      import chisel3._
      import chisel3.properties.Property
      Property[Seq[Double]]().contains(Property[Double]())
    """)
    assertTypeError("""
      import chisel3._
      import chisel3.properties.Property
      Property[Seq[String]]().contains("cold")
    """)
  }

  // TODO: Restore this success smoke test once nightly CIRCT is bumped to support list_contains.
  it should "fail to emit SystemVerilog when firtool does not support list_contains" in {
    val exception = intercept[RuntimeException] {
      ChiselStage.emitSystemVerilog(new RawModule {
        val names = IO(Input(Property[Seq[String]]()))
        val target = IO(Input(Property[String]()))
        val found = IO(Output(Property[Boolean]()))
        found := names.contains(target)
      })
    }
    exception.getMessage should include("use of unknown declaration 'list_contains'")
  }
}
