// SPDX-License-Identifier: Apache-2.0

package chiselTests.properties

import chisel3._
import chisel3.properties.{Class, Property}
import chisel3.experimental.hierarchy.Definition
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
}
