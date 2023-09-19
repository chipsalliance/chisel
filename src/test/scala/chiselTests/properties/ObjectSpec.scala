// SPDX-License-Identifier: Apache-2.0

package chiselTests.properties

import chisel3._
import chisel3.properties.{Class, Property}
import chisel3.experimental.hierarchy.{Definition, Instance}
import chiselTests.{ChiselFlatSpec, MatchesAndOmits}
import circt.stage.ChiselStage

class ObjectSpec extends ChiselFlatSpec with MatchesAndOmits {
  behavior.of("DynamicObject")

  it should "support Objects in Class ports" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
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
    })

    matchesAndOmits(chirrtl)(
      "class Parent",
      "output out : Inst<Test>",
      "propassign out, obj1"
    )()
  }

  it should "support Objects in Module ports" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
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
    })

    matchesAndOmits(chirrtl)(
      "module Parent",
      "output out : Inst<Test>",
      "propassign out, obj1"
    )()
  }

  it should "support output Object fields as sources" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val cls = Definition(new Class {
        override def desiredName = "Test"
        val out = IO(Output(Property[Int]()))
      })

      val out = IO(Output(Property[Int]()))
      val obj1 = Class.unsafeGetDynamicObject("Test")
      out := obj1.getField[Int]("out")
    })

    matchesAndOmits(chirrtl)(
      "object obj1 of Test",
      "propassign out, obj1.out"
    )()
  }

  it should "support input Object fields as sinks" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val cls = Definition(new Class {
        override def desiredName = "Test"
        val in = IO(Input(Property[Int]()))
      })

      val in = IO(Input(Property[Int]()))
      val obj1 = Class.unsafeGetDynamicObject("Test")
      obj1.getField[Int]("in") := in
    })

    matchesAndOmits(chirrtl)(
      "object obj1 of Test",
      "propassign obj1.in, in"
    )()
  }

  behavior.of("StaticObject")

  it should "support Instances of Objects in Class ports" in {
    val chirrtl = ChiselStage.emitCHIRRTL(
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
            val out1 = IO(Output(Class.getReferenceType(cls1)))
            val out2 = IO(Output(Class.getReferenceType(cls2)))

            val obj1 = Instance(cls1)
            val obj2 = Instance(cls2)

            out1 := obj1
            out2 := obj2
          }
        })
      }
    )

    matchesAndOmits(chirrtl)(
      "class Parent",
      "output out1 : Inst<Test>",
      "output out2 : Inst<Test_1>",
      "object obj1 of Test",
      "object obj2 of Test_1",
      "propassign out1, obj1",
      "propassign out2, obj2"
    )()
  }

  it should "support Instances of Objects in Module ports" in {
    val chirrtl = ChiselStage.emitCHIRRTL(
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
          val out1 = IO(Output(Class.getReferenceType(cls1)))
          val out2 = IO(Output(Class.getReferenceType(cls2)))

          val obj1 = Instance(cls1)
          val obj2 = Instance(cls2)

          out1 := obj1
          out2 := obj2
        })
      }
    )

    matchesAndOmits(chirrtl)(
      "module Parent",
      "output out1 : Inst<Test>",
      "output out2 : Inst<Test_1>",
      "object obj1 of Test",
      "object obj2 of Test_1",
      "propassign out1, obj1",
      "propassign out2, obj2"
    )()
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
          val outClass1 = IO(Output(Class.getReferenceType(cls1)))
          val objClass2 = Instance(cls2)
          outClass1 := objClass2
        })
      }
    )

    e.getMessage should include(
      "Sink Property[ClassType] expected class Test, but source Instance[Class] was class Test_1"
    )
  }
}
