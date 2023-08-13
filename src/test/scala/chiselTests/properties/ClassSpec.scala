// SPDX-License-Identifier: Apache-2.0

package chiselTests.properties

import chisel3._
import chisel3.properties.{Class, Property}
import chiselTests.{ChiselFlatSpec, MatchesAndOmits}
import circt.stage.ChiselStage

class ClassSpec extends ChiselFlatSpec with MatchesAndOmits {
  behavior.of("Class")

  it should "serialize to FIRRTL with anonymous names" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      Module(new Class)
      Module(new Class)
      Module(new Class)
    })

    matchesAndOmits(chirrtl)(
      "class Class",
      "class Class_1",
      "class Class_2"
    )()
  }

  it should "serialize to FIRRTL with a desiredName" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      Module(new Class {
        override def desiredName = "Foo"
      })
    })

    matchesAndOmits(chirrtl)(
      "class Foo"
    )()
  }

  it should "serialize to FIRRTL with the Scala class name" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      class MyClass extends Class {}

      Module(new MyClass)
    })

    matchesAndOmits(chirrtl)(
      "class MyClass"
    )()
  }

  it should "support Property type ports" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      Module(new Class {
        val in = IO(Input(Property[Int]()))
        val out = IO(Output(Property[Int]()))
      })
    })

    matchesAndOmits(chirrtl)(
      "input in : Integer",
      "output out : Integer"
    )()
  }

  it should "only support Property type ports" in {
    (the[ChiselException] thrownBy {
      ChiselStage.emitCHIRRTL(new RawModule {
        Module(new Class {
          val in = IO(Input(Bool()))
        })
      })
    }).getMessage should include("Class ports must be Property type")
  }

  it should "support Property assignments" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      Module(new Class {
        val in = IO(Input(Property[Int]()))
        val out = IO(Output(Property[Int]()))
        out := in
      })
    })

    matchesAndOmits(chirrtl)(
      "propassign out, in"
    )()
  }

  it should "support instantiation through its own API" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val cls = Module(new Class {
        override def desiredName = "Test"
        val in = IO(Input(Property[Int]()))
        val out = IO(Output(Property[Int]()))
        out := in
      })

      val obj1 = cls.instantiate
      val obj2 = cls.instantiate
    })

    matchesAndOmits(chirrtl)(
      "class Test",
      "object obj1",
      "object obj2"
    )()
  }

  it should "support instantiation within a Class" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val cls = Module(new Class {
        override def desiredName = "Test"
        val in = IO(Input(Property[Int]()))
        val out = IO(Output(Property[Int]()))
        out := in
      })

      Module(new Class {
        override def desiredName = "Parent"
        val obj1 = cls.instantiate
      })
    })

    matchesAndOmits(chirrtl)(
      "class Test",
      "class Parent",
      "object obj1"
    )()
  }

  it should "provide a static method to materialize a Property[Class] type from a name" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val io = IO(Input(Class.getReferenceType("foo").asInstanceOf[Property[Class]]))
    })
    println(chirrtl)
    matchesAndOmits(chirrtl)(
      "input io : Inst<foo>"
    )()
  }
}
