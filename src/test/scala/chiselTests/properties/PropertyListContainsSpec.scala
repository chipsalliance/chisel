// SPDX-License-Identifier: Apache-2.0

package chiselTests.properties

import chisel3._
import chisel3.domains.ClockDomain
import chisel3.experimental.hierarchy.Definition
import chisel3.properties.{AnyClassType, Class, Property}
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

  private class ReferenceContainsModule extends RawModule {
    private val cls = Definition(new Class)
    private val clsType = cls.getClassType

    val objects = IO(Input(Property[Seq[clsType.Type]]()))
    val refs = IO(Input(Property[List[AnyClassType]]()))

    val obj = IO(Input(cls.getPropertyType))
    val ref = IO(Input(Property[AnyClassType]()))
    val domain = IO(Input(ClockDomain.Type()))

    val objectFound = IO(Output(Property[Boolean]()))
    val refFound = IO(Output(Property[Boolean]()))
    val instInAny = IO(Output(Property[Boolean]()))
    val domainInAny = IO(Output(Property[Boolean]()))

    objectFound := objects.contains(obj)
    refFound := refs.contains(ref)
    instInAny := refs.contains(obj)
    domainInAny := refs.contains(domain)
  }

  it should "support class-specific and AnyRef sequences" in {
    ChiselStage
      .emitCHIRRTL(new ReferenceContainsModule)
      .fileCheck() {
        """|CHECK: list_contains(objects, obj)
           |CHECK: list_contains(refs, ref)
           |CHECK: list_contains(refs, obj)
           |CHECK: list_contains(refs, domain)
           |""".stripMargin
      }
  }

  private class Foo extends Class
  private class Bar extends Class

  private def assertContainsError(expectedMessage: String)(module: => RawModule): Unit = {
    val exception = intercept[ChiselException] {
      ChiselStage.emitCHIRRTL(module, Array("--throw-on-first-error"))
    }
    exception.getMessage should include(expectedMessage)
  }

  it should "reject a different-class search value in a class-specific sequence" in {
    assertContainsError("list_contains element type Inst<Bar> does not match list element type Inst<Foo>") {
      new RawModule {
        private val foo = Definition(new Foo)
        private val bar = Definition(new Bar)
        private val fooType = foo.getClassType

        val objects = IO(Input(Property[Seq[fooType.Type]]()))
        val target = IO(Input(bar.getPropertyType))
        objects.contains(target)
      }
    }
  }

  it should "reject an AnyRef search value in a class-specific sequence" in {
    assertContainsError("list_contains element type AnyRef does not match list element type Inst<Foo>") {
      new RawModule {
        private val foo = Definition(new Foo)
        private val fooType = foo.getClassType

        val objects = IO(Input(Property[Seq[fooType.Type]]()))
        val target = IO(Input(Property[AnyClassType]()))
        objects.contains(target)
      }
    }
  }

  it should "reject a domain search value in a class-specific sequence" in {
    assertContainsError("list_contains with a domain reference requires list element type AnyRef, but got Inst<Foo>") {
      new RawModule {
        private val foo = Definition(new Foo)
        private val fooType = foo.getClassType

        val objects = IO(Input(Property[Seq[fooType.Type]]()))
        val target = IO(Input(ClockDomain.Type()))
        objects.contains(target)
      }
    }
  }

  it should "reject reference/scalar membership tests" in {
    assertTypeError("Property[Seq[AnyClassType]]().contains(Property[Int]())")
    assertTypeError("Property[Seq[Int]]().contains(Property[AnyClassType]())")
    assertTypeError("Property[Seq[Int]]().contains(ClockDomain.Type())")
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
    assertTypeError("Property[AnyClassType]().contains(ClockDomain.Type())")
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

  // TODO: Restore this success smoke test once firtool supports list_contains with class and domain references.
  it should "fail to emit SystemVerilog when firtool does not support reference list_contains" in {
    val exception = intercept[RuntimeException] {
      ChiselStage.emitSystemVerilog(new ReferenceContainsModule)
    }
    exception.getMessage should include("list_contains does not support elements of type")
  }

  it should "compile with firtool" in {
    ChiselStage.emitSystemVerilog(new RawModule {
      val names = IO(Input(Property[Seq[String]]()))
      val target = IO(Input(Property[String]()))
      val found = IO(Output(Property[Boolean]()))
      found := names.contains(target)
    })
  }
}
