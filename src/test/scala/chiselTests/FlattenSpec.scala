// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.experimental.dedupGroup
import chisel3.testing.FileCheck
import chisel3.util.experimental.{FlattenInstance, FlattenInstanceAllowDedup}
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

object FlattenSpec {
  trait SimpleIO {
    val a = IO(Input(Bool()))
    val b = IO(Output(Bool()))
  }

  class Leaf extends RawModule with SimpleIO {
    b := ~a
  }

  class Middle extends RawModule with SimpleIO {
    val leaf = Module(new Leaf)
    leaf.a := a
    b := leaf.b
  }

  class Parent extends RawModule with SimpleIO {
    // Workaround for bug in anonymous module naming.
    override def desiredName: String = "Parent"
    val middle = Module(new Middle)
    middle.a := a
    b := middle.b
  }
}

class FlattenSpec extends AnyFlatSpec with Matchers with FileCheck {
  import FlattenSpec._

  "FlattenInstance" should "Flatten only the instance marked with FlattenInstance and its children" in {
    class Top extends RawModule with SimpleIO {
      val notFlat = Module(new Parent)
      val flat = Module(new Parent with FlattenInstance)
      notFlat.a := a
      flat.a := a
      b := notFlat.b ^ flat.b
    }
    ChiselStage
      .emitSystemVerilog(new Top)
      .fileCheck("--implicit-check-not={{^ *}}module")(
        """|CHECK: module Leaf
           |CHECK: module Middle
           |CHECK: module Parent
           |CHECK: module Parent_1
           |CHECK: module Top
           |""".stripMargin
      )
  }

  "FlattenInstanceAllowDedup" should "Flatten any module that dedups with a module marked flatten" in {
    class Top extends RawModule with SimpleIO {
      val notFlat = Module(new Parent)
      val flat = Module(new Parent with FlattenInstanceAllowDedup)
      notFlat.a := a
      flat.a := a
      b := notFlat.b ^ flat.b
    }
    ChiselStage
      .emitSystemVerilog(new Top)
      .fileCheck("--implicit-check-not={{^ *}}module")(
        """|CHECK: module Parent
           |CHECK: module Top
           |""".stripMargin
      )
  }
}
