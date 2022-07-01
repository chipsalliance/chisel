// SPDX-License-Identifier: Apache-2.0

package chiselTests.naming

import chisel3._
import chisel3.aop.Select
import chisel3.stage.{ChiselGeneratorAnnotation, ChiselStage}
import chisel3.experimental.{dump, noPrefix, prefix, treedump}
import chiselTests.{ChiselPropSpec, Utils}
import chisel3.stage.{DesignAnnotation, NoRunFirrtlCompilerAnnotation}
import firrtl.options.{Dependency, Phase, PhaseManager}
import firrtl.options.phases.DeletedWrapper

class SuggestNameSpec extends ChiselPropSpec with Utils {
  implicit val minimumMajorVersion: Int = 12

  property("0. Calling suggestName 2x should be a runtime deprecation") {
    class Test extends Module {
      {
        val wire = {
          val x = WireInit(0.U(3.W)).suggestName("mywire")
          dontTouch(x)
        }
        wire.suggestName("somethingElse")
      }
    }
    val (log, _) = grabLog(ChiselStage.emitVerilog(new Test()))
    log should include(
      "Calling suggestName (somethingElse, when already called with Some(mywire)) will become an error in Chisel 3.6"
    )
  }

  property("1. Calling suggestName outside of a Builder context should be an error") {
    class Test extends Module {
      val wire = {
        val x = WireInit(0.U(3.W))
        dontTouch(x)
      }
    }

    val pm = new PhaseManager(Seq(Dependency[chisel3.stage.phases.Checks], Dependency[chisel3.stage.phases.Elaborate]))
    val test = pm
      .transform(Seq(ChiselGeneratorAnnotation(() => new Test()), NoRunFirrtlCompilerAnnotation))
      .collectFirst {
        case d: DesignAnnotation[_] => d
      }
      .get
      .design
    val caught = intercept[IllegalArgumentException] {
      test.asInstanceOf[Test].wire.suggestName("somethingElse")
    }
    caught.getMessage should include("suggestName (somethingElse) should only be called from a Builder context")
  }

  property("2. Calling suggestName after module close should be a runtime deprecation") {
    class Child extends Module {
      val wire = {
        val x = WireInit(0.U(3.W))
        dontTouch(x)
      }
    }
    class Test extends Module {
      val child = Module(new Child())
      child.wire.suggestName("somethingElse")
    }
    val (log, _) = grabLog(ChiselStage.emitVerilog(new Test()))
    log should include(
      "Calling suggestName (somethingElse, on Child.wire: Wire[UInt<3>], when the containing module (Child) completed elaboration already"
    )
  }

  property("3. Calling suggestName with the same thing prefix would have given should be a runtime deprecation") {
    class Test extends Module {
      val wire = {
        val x = WireInit(0.U(3.W)).suggestName("wire")
        dontTouch(x)
      }
    }
    val (log, _) = grabLog(ChiselStage.emitVerilog(new Test()))
    log should include(
      "calling suggestName(wire) had no effect as it is the same as the auto prefixed name"
    )
  }

  property("5a. Calling suggestName on a node should be allowed") {
    class Example extends Module {
      val foo, bar = IO(Input(UInt(8.W)))
      val out = IO(Output(UInt(8.W)))

      val sum = foo +& bar
      sum.suggestName("fuzz")
      out := sum
    }
    val (log, chirrtl) = grabLog(ChiselStage.emitChirrtl(new Example))
    log should equal("")
    chirrtl should include("node fuzz = add(foo, bar)")
  }

  property("5a. Calling suggestName on an IO should be allowed") {
    class Example extends Module {
      val foo, bar = IO(Input(UInt(8.W)))
      val out = IO(Output(UInt(8.W)))

      val sum = foo +& bar
      foo.suggestName("FOO")
      bar.suggestName("BAR")
      out := sum
    }
    val (log, chirrtl) = grabLog(ChiselStage.emitChirrtl(new Example))
    log should equal("")
    chirrtl should include("node sum = add(FOO, BAR)")
  }

  property("5b. Calling suggestName on a prefixed node should be allowed") {
    class Example extends Module {
      val foo, bar = IO(Input(UInt(8.W)))
      val out = IO(Output(UInt(8.W)))

      out := {
        val sum = {
          val node = foo +& bar
          node.suggestName("fuzz")
          node +% 0.U
        }
        sum
      }
    }
    val (log, chirrtl) = grabLog(ChiselStage.emitChirrtl(new Example))
    log should equal("")
    chirrtl should include("node out_sum_fuzz = add(foo, bar)")
  }

  property("5c. Calling suggestName on a Module instance should be allowed") {
    import chisel3.util._

    class PassThrough extends Module {
      val enq = IO(Flipped(Decoupled(UInt(8.W))))
      val deq = IO(Decoupled(UInt(8.W)))
      deq <> enq
    }

    class Example extends Module {
      val enq = IO(Flipped(Decoupled(UInt(8.W))))
      val deq = IO(Decoupled(UInt(8.W)))

      val q = Module(new PassThrough())
      q.enq <> enq
      deq <> q.deq
      q.suggestName("fuzz")
    }

    val (log, chirrtl) = grabLog(ChiselStage.emitChirrtl(new Example))
    log should equal("")
    chirrtl should include("inst fuzz of PassThrough")
  }

  /*
  // This test is commented out until https://github.com/chipsalliance/chisel3/issues/2613 is resolved
  property("5d. Calling suggestName on an Instance instance should be allowed") {
    import chisel3.experimental.hierarchy.{Definition, Instance}
    import chiselTests.experimental.hierarchy.Examples.AddOne
    class Example extends Module {
      val defn = Definition(new AddOne)
      val inst = Instance(defn)
      inst.suggestName("fuzz")
      val fuzz = inst
    }
    val (log, chirrtl) = grabLog(ChiselStage.emitChirrtl(new Example))
    log should equal("")
    chirrtl should include("inst fuzz of AddOne")
  }
   */

  property("5e. Calling suggestName on a Mem should be allowed") {
    class Example extends Module {
      val mem = SyncReadMem(8, UInt(8.W))

      mem.suggestName("fuzz")
    }
    val (log, chirrtl) = grabLog(ChiselStage.emitChirrtl(new Example))
    log should equal("")
    chirrtl should include("smem fuzz")
  }

  property("5f. Calling suggestName on a verif statement should be a runtime deprecation") {
    class Example extends Module {
      val in = IO(Input(UInt(8.W)))
      val z = chisel3.assert(in =/= 123.U)
      z.suggestName("fuzz")
    }
    val (log, chirrtl) = grabLog(ChiselStage.emitChirrtl(new Example))
    log should include("Calling suggestName (fuzz, on something that cannot actually be named: chisel3.assert$Assert")
    (chirrtl should include).regex("assert.*: z")
  }

  property("5g. Calling suggestName on a literal should be a runtime deprecation") {
    class Example extends Module {
      val out = IO(Output(UInt(8.W)))

      val sum = 0.U
      sum.suggestName("fuzz")
      out := sum
    }
    val (log, chirrtl) = grabLog(ChiselStage.emitChirrtl(new Example))
    log should include("Calling suggestName (fuzz, on something that cannot actually be named: UInt<1>(0)")
    chirrtl should include("out <= UInt")
  }

  property("5h. Calling suggestName on a field of an Aggregate should be a runtime deprecation") {
    class Example extends Module {
      val io = IO(new Bundle {
        val in = Input(UInt(8.W))
        val out = Output(UInt(8.W))
      })
      io.in.suggestName("fuzz")
      io.out.suggestName("bar")
      io.out := io.in
    }
    val (log, chirrtl) = grabLog(ChiselStage.emitChirrtl(new Example))
    log should include("Calling suggestName (fuzz, on something that cannot actually be named: Example.io.in")
    log should include("Calling suggestName (bar, on something that cannot actually be named: Example.io.out")

    chirrtl should include("io.out <= io.in")
  }

  property("5i. Calling suggestName on unbound Data should be a runtime deprecation") {
    class Example extends Module {
      val in = IO(Input(UInt(8.W)))
      val out = IO(Output(UInt(8.W)))
      val z = UInt(8.W)
      z.suggestName("fuzz")
      out := in
    }
    val (log, chirrtl) = grabLog(ChiselStage.emitChirrtl(new Example))
    log should include("Calling suggestName (fuzz, on something that cannot actually be named: UInt<8>)")
    chirrtl should include("out <= in")
  }
}
