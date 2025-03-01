// SPDX-License-Identifier: Apache-2.0

package chiselTests.properties

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Definition, Instance}
import chisel3.properties.{Class, DynamicObject, Property}
import chisel3.testing.scalatest.FileCheck
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ClassSpec extends AnyFlatSpec with Matchers with FileCheck {
  behavior.of("Class")

  it should "serialize to FIRRTL with anonymous names" in {
    ChiselStage.emitCHIRRTL {
      new RawModule {
        Definition(new Class)
        Definition(new Class)
        Definition(new Class)
      }
    }.fileCheck()(
      """|CHECK: class Class
         |CHECK: class Class_1
         |CHECK: class Class_2
         |""".stripMargin
    )
  }

  it should "serialize to FIRRTL with a desiredName" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      Definition(new Class {
        override def desiredName = "Foo"
      })
    }) should include("class Foo")
  }

  it should "serialize to FIRRTL with the Scala class name" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      class MyClass extends Class {}

      Definition(new MyClass)
    }) should include("class MyClass")
  }

  it should "support Property type ports" in {
    ChiselStage.emitCHIRRTL {
      new RawModule {
        Definition(new Class {
          val in = IO(Input(Property[Int]()))
          val out = IO(Output(Property[Int]()))
        })
      }
    }.fileCheck()(
      """|CHECK: input in : Integer
         |CHECK: output out : Integer
         |""".stripMargin
    )
  }

  it should "only support Property type ports" in {
    val e = the[ChiselException] thrownBy {
      ChiselStage.emitCHIRRTL(
        new RawModule {
          Definition(new Class {
            val in = IO(Input(Bool()))
          })
        },
        Array("--throw-on-first-error")
      )
    }
    e.getMessage should include("Class ports must be Property type, but found Bool.")
  }

  it should "support Property assignments" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      Definition(new Class {
        val in = IO(Input(Property[Int]()))
        val out = IO(Output(Property[Int]()))
        out := in
      })
    }) should include("propassign out, in")
  }

  it should "support instantiation through its own API" in {
    ChiselStage.emitCHIRRTL {
      new RawModule {
        val cls = Definition(new Class {
          override def desiredName = "Test"
          val in = IO(Input(Property[Int]()))
          val out = IO(Output(Property[Int]()))
          out := in
        })

        val obj1 = Class.unsafeGetDynamicObject("Test")
        val obj2 = Class.unsafeGetDynamicObject("Test")
      }
    }.fileCheck()(
      """|CHECK: class Test
         |CHECK: object obj1 of Test
         |CHECK: object obj2 of Test
         |""".stripMargin
    )
  }

  it should "support instantiation within a Class" in {
    ChiselStage.emitCHIRRTL {
      new RawModule {
        val cls = Definition(new Class {
          override def desiredName = "Test"
          val in = IO(Input(Property[Int]()))
          val out = IO(Output(Property[Int]()))
          out := in
        })

        Definition(new Class {
          override def desiredName = "Parent"
          val obj1 = Class.unsafeGetDynamicObject("Test")
        })
      }
    }.fileCheck()(
      """|CHECK: class Test
         |CHECK: class Parent
         |CHECK: object obj1 of Test
         |""".stripMargin
    )
  }

  it should "support instantiation with Instance" in {
    ChiselStage.emitCHIRRTL {
      new RawModule {
        val cls = Definition(new Class {
          override def desiredName = "Test"
          val in = IO(Input(Property[Int]()))
          val out = IO(Output(Property[Int]()))
          out := in
        })

        val obj1 = Instance(cls)
        val obj2 = Instance(cls)
      }
    }.fileCheck()(
      """|CHECK:         class Test
         |CHECK:         object obj1 of Test
         |CHECK:         object obj2 of Test
         |""".stripMargin
    )
  }

  it should "support @instantiable and @public" in {
    ChiselStage.emitCHIRRTL {
      @instantiable
      class Test extends Class {
        @public val in = IO(Input(Property[Int]()))
        @public val out = IO(Output(Property[Int]()))
        out := in
      }

      new RawModule {
        val cls = Definition(new Test)

        val obj1 = Instance(cls)
        val obj2 = Instance(cls)

        obj2.in := obj1.out
      }
    }.fileCheck()(
      """|CHECK-LABEL: class Test
         |CHECK-LABEL: public module
         |CHECK:         object obj1 of Test
         |CHECK:         object obj2 of Test
         |CHECK:         propassign obj2.in, obj1.out
         |""".stripMargin
    )
  }

  it should "support getting a ClassType from a Class Definition" in {
    class Test extends Class {}

    ChiselStage.emitCHIRRTL(new RawModule {
      val definition = Definition(new Test)
      val classType = definition.getClassType
      classType.name should equal("Test")
    })
  }
}
