// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.stage.ChiselStage

import org.scalatest._
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.should.Matchers

class LiteralToTargetSpec extends AnyFreeSpec with Matchers {

  "Literal Data should fail to be converted to ReferenceTarget" in {

    (the[chisel3.internal.ChiselException] thrownBy {

      class Bar extends RawModule {
        val a = 1.U
      }

      class Foo extends RawModule {
        val bar = Module(new Bar)
        bar.a.toTarget
      }

      ChiselStage.elaborate(new Foo)
    } should have).message("Illegal component name: UInt<1>(\"h01\") (note: literals are illegal)")
  }

  "Vectors which are indexed by literal values should successfully be converted to Reference Target" in {
    class Bar extends RawModule {
      val vec = IO(Input(Vec(3, Bool())))
      val lit = 1.U
      val notLit = vec(lit)
    }
    class Foo extends RawModule {
      val bar = Module(new Bar)
      bar.notLit.toTarget
    }
    // This does not work currently
    ChiselStage.elaborate(new Foo)
  }

  "Vectors which are accesesd with Scala indices or iterated over should successfully be converted to Reference Target" in {
    class Bar extends RawModule {
      val vecIn = IO(Input(Vec(3, Bool())))
      val vecOut = IO(Output(Vec(3, Bool())))
      vecOut := vecIn
    }
    class Foo extends RawModule {
      val bar = Module(new Bar)
      bar.vecIn.map(_.toTarget)
      bar.vecIn(0).toTarge
    }
    ChiselStage.elaborate(new Foo)
  }

  "Should not be able to annotate the result of a vector lookup" in {

    class Bar extends RawModule {
      val vec = IO(Input(Vec(8, Bool())))
      val sel = IO(Input(UInt(3.W)))
      val out = IO(Output(Bool()))
      val tmp = vec(sel)
      out := tmp
    }
    class Foo extends RawModule {
      val bar = Module(new Bar)
      bar.tmp.toTarget
    }
    // assert this fails
    ChiselStage.elaborate(new Foo)
  }
  "Should  be able to annotate the result of a vector lookup if assigned to a temporary" in {

    class Bar extends RawModule {
      val vec = IO(Input(Vec(8, Bool())))
      val sel = IO(Input(UInt(3.W)))
      val out = IO(Output(Bool()))
      val tmp = WireInit(vec(sel))
      out := tmp
    }
    class Foo extends RawModule {
      val bar = Module(new Bar)
      bar.tmp.toTarget
    }
    ChiselStage.elaborate(new Foo)
  }
}
