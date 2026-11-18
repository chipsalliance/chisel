// SPDX-License-Identifier: Apache-2.0

package chiselTests.properties

import chisel3._
import chisel3.experimental.hierarchy.Definition
import chisel3.properties.{AnyClassType, Class, Property}
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

  private class ReferenceEqualityModule extends RawModule {
    val foo = Definition(new Class)
    val bar = Definition(new Class)
    val a = IO(Input(foo.getPropertyType))
    val b = IO(Input(bar.getPropertyType))
    val ref = IO(Input(Property[AnyClassType]()))
    val sameClass = IO(Output(Property[Boolean]()))
    val differentClasses = IO(Output(Property[Boolean]()))
    val instAny = IO(Output(Property[Boolean]()))
    val anyInst = IO(Output(Property[Boolean]()))
    val anyAny = IO(Output(Property[Boolean]()))
    sameClass := a === a
    differentClasses := a === b
    instAny := a === ref
    anyInst := ref === a
    anyAny := ref === ref
  }

  it should "compare Inst and AnyRef properties in all supported combinations" in {
    ChiselStage
      .emitCHIRRTL(new ReferenceEqualityModule)
      .fileCheck() {
        """|CHECK: propassign [[SAME_CLASS:[^,]+]], prop_eq(a, a)
           |CHECK: propassign sameClass, [[SAME_CLASS]]
           |
           |CHECK: propassign [[DIFFERENT_CLASSES:[^,]+]], prop_eq(a, b)
           |CHECK: propassign differentClasses, [[DIFFERENT_CLASSES]]
           |
           |CHECK: propassign [[INST_ANY:[^,]+]], prop_eq(a, ref)
           |CHECK: propassign instAny, [[INST_ANY]]
           |
           |CHECK: propassign [[ANY_INST:[^,]+]], prop_eq(ref, a)
           |CHECK: propassign anyInst, [[ANY_INST]]
           |
           |CHECK: propassign [[ANY_ANY:[^,]+]], prop_eq(ref, ref)
           |CHECK: propassign anyAny, [[ANY_ANY]]
           |""".stripMargin
      }
  }

  // TODO: Restore this success smoke test once firtool is bumped to support reference equality.
  it should "fail to emit SystemVerilog when firtool does not support reference equality" in {
    val exception = intercept[RuntimeException] {
      ChiselStage.emitSystemVerilog(new ReferenceEqualityModule)
    }
    exception.getMessage should include("lhs of prop_eq must be String, Bool, or Integer type")
  }

  it should "reject reference/scalar comparisons" in {
    assertTypeError("""val x: Property[Boolean] = Class.unsafeGetReferenceType("Foo") === Property[Int]()""")
    assertTypeError("""val x: Property[Boolean] = Property[Int]() === Class.unsafeGetReferenceType("Foo")""")
    assertTypeError("val x: Property[Boolean] = Property[AnyClassType]() === Property[Boolean]()")
    assertTypeError("val x: Property[Boolean] = Property[Boolean]() === Property[AnyClassType]()")
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
