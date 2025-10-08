// SPDX-License-Identifier: Apache-2.0

package chiselTests.properties

import chisel3._
import chisel3.experimental.hierarchy.{Definition, Instance}
import chisel3.properties.{Class, DynamicObject, Property}
import chisel3.testing.scalatest.FileCheck
import chisel3.util.experimental.BoringUtils
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ObjectSpec extends AnyFlatSpec with Matchers with FileCheck {
  behavior.of("DynamicObject")

  it should "support Objects in Class ports" in {
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
          val out = IO(Output(Class.unsafeGetReferenceType("Test")))
          val obj1 = Class.unsafeGetDynamicObject("Test")
          out := obj1.getReference
        })
      }
    }.fileCheck()(
      """|CHECK: class Parent
         |CHECK: output out : Inst<Test>
         |CHECK: propassign out, obj1
         |""".stripMargin
    )
  }

  it should "support Objects in Module ports" in {
    ChiselStage.emitCHIRRTL {
      new RawModule {
        val cls = Definition(new Class {
          override def desiredName = "Test"
          val in = IO(Input(Property[Int]()))
          val out = IO(Output(Property[Int]()))
          out := in
        })

        Module(new RawModule {
          override def desiredName = "Parent"
          val out = IO(Output(Class.unsafeGetReferenceType("Test")))
          val obj1 = Class.unsafeGetDynamicObject("Test")
          out := obj1.getReference
        })
      }
    }.fileCheck()(
      """|CHECK: module Parent
         |CHECK: output out : Inst<Test>
         |CHECK: propassign out, obj1
         |""".stripMargin
    )
  }

  it should "support output Object fields as sources" in {
    ChiselStage.emitCHIRRTL {
      new RawModule {
        val cls = Definition(new Class {
          override def desiredName = "Test"
          val out = IO(Output(Property[Int]()))
        })

        val out = IO(Output(Property[Int]()))
        val obj1 = Class.unsafeGetDynamicObject("Test")
        out := obj1.getField[Int]("out")
      }
    }.fileCheck()(
      """|CHECK: object obj1 of Test
         |CHECK: propassign out, obj1.out
         |""".stripMargin
    )
  }

  it should "support input Object fields as sinks" in {
    ChiselStage.emitCHIRRTL {
      new RawModule {
        val cls = Definition(new Class {
          override def desiredName = "Test"
          val in = IO(Input(Property[Int]()))
        })

        val in = IO(Input(Property[Int]()))
        val obj1 = Class.unsafeGetDynamicObject("Test")
        obj1.getField[Int]("in") := in
      }
    }.fileCheck()(
      """|CHECK: object obj1 of Test
         |CHECK: propassign obj1.in, in
         |""".stripMargin
    )
  }

  it should "support creating DynamicObject from a Class with DynamicObject.apply" in {
    ChiselStage.emitCHIRRTL {
      new RawModule {
        val out = IO(Output(Property[Int]()))

        val obj = DynamicObject(new Class {
          override def desiredName = "Test"
          val in = IO(Input(Property[Int]()))
          val out = IO(Output(Property[Int]()))
        })

        obj.getField[Int]("in") := Property(1)

        out := obj.getField[Int]("out")
      }
    }.fileCheck()(
      """|CHECK: object obj of Test
         |CHECK: propassign obj.in, Integer(1)
         |CHECK: propassign out, obj.out
         |""".stripMargin
    )
  }

  it should "support boring ports through a Class created with DynamicObject.apply" in {
    ChiselStage.emitCHIRRTL {
      new RawModule {
        val in = IO(Input(Property[Int]()))

        val obj = DynamicObject(new Class {
          override def desiredName = "Test"
          val out = IO(Output(Property[Int]()))
          out := BoringUtils.bore(in)
        })
      }
    }.fileCheck()(
      """|CHECK: input out_bore : Integer
         |CHECK: propassign out, out_bore
         |CHECK: propassign obj.out_bore, in
         |""".stripMargin
    )
  }

  behavior.of("StaticObject")

  it should "support Instances of Objects in Class ports" in {
    ChiselStage.emitCHIRRTL {
      new RawModule {
        Definition({
          val cls1 = Definition(new Class {
            override def desiredName = "Test"
            val in = IO(Input(Property[Int]()))
            val out = IO(Output(Property[Int]()))
            out := in
          })

          val cls2 = Definition(new Class {
            override def desiredName = "Test"
            val in = IO(Input(Property[Int]()))
            val out = IO(Output(Property[Int]()))
            out := in
          })

          new Class {
            override def desiredName = "Parent"
            val out1 = IO(Output(cls1.getPropertyType))
            val out2 = IO(Output(cls2.getPropertyType))

            val obj1 = Instance(cls1)
            val obj2 = Instance(cls2)

            out1 := obj1.getPropertyReference
            out2 := obj2.getPropertyReference
          }
        })
      }
    }.fileCheck()(
      """|CHECK: class Parent
         |CHECK: output out1 : Inst<Test>
         |CHECK: output out2 : Inst<Test_1>
         |CHECK: object obj1 of Test
         |CHECK: object obj2 of Test_1
         |CHECK: propassign out1, obj1
         |CHECK: propassign out2, obj2
         |""".stripMargin
    )
  }

  it should "support Instances of Objects in Module ports" in {
    ChiselStage.emitCHIRRTL {
      new RawModule {
        val cls1 = Definition(new Class {
          override def desiredName = "Test"
          val in = IO(Input(Property[Int]()))
          val out = IO(Output(Property[Int]()))
          out := in
        })
        val cls2 = Definition(new Class {
          override def desiredName = "Test"
          val in = IO(Input(Property[Int]()))
          val out = IO(Output(Property[Int]()))
          out := in
        })

        Module(new RawModule {
          override def desiredName = "Parent"
          val out1 = IO(Output(cls1.getPropertyType))
          val out2 = IO(Output(cls2.getPropertyType))

          val obj1 = Instance(cls1)
          val obj2 = Instance(cls2)

          out1 := obj1.getPropertyReference
          out2 := obj2.getPropertyReference
        })
      }
    }.fileCheck()(
      """|CHECK: module Parent
         |CHECK: output out1 : Inst<Test>
         |CHECK: output out2 : Inst<Test_1>
         |CHECK: object obj1 of Test
         |CHECK: object obj2 of Test_1
         |CHECK: propassign out1, obj1
         |CHECK: propassign out2, obj2
         |""".stripMargin
    )
  }

  it should "error for Instances of Objects in Module ports of the wrong type" in {
    val e = the[ChiselException] thrownBy ChiselStage.emitCHIRRTL(
      new RawModule {
        val cls1 = Definition(new Class {
          override def desiredName = "Test"
        })
        val cls2 = Definition(new Class {
          override def desiredName = "Test"
        })

        Module(new RawModule {
          val outClass1 = IO(Output(cls1.getPropertyType))
          val objClass2 = Instance(cls2)
          outClass1 := objClass2.getPropertyReference
        })
      }
    )

    e.getMessage should include(
      "Sink Property[ClassType] expected class Test, but source Instance[Class] was class Test_1"
    )
  }
}
