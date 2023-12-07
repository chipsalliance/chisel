// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.probe._
import chisel3.util.Counter
import chisel3.testers.{BasicTester, TesterDriver}
import circt.stage.ChiselStage
import chisel3.util.DecoupledIO
import chisel3.reflect.DataMirror
import firrtl.transforms.DontTouchAnnotation
import chisel3.experimental.ExtModule

class DebugSpec extends ChiselFlatSpec with MatchesAndOmits {
  private def pruneSourceLoc(s: String): String = {
    s.split("\n").map(_.takeWhile(_ != '@')).mkString("\n")
  }
  class DecoupledAgg extends Bundle {
    val incoming = Flipped(DecoupledIO(UInt(8.W)))
    val outgoing = DecoupledIO(UInt(8.W))
  }

  "Debug kinds" should "work in ports and produce valid define" in {
    class Example extends Module {
      val a = IO(Flipped(new DecoupledAgg()))

      val take = IO(Debug.takeOver(new DecoupledAgg()))
      val prod = IO(Debug.producer(new DecoupledAgg()))
      val cons = IO(Debug.consumer(new DecoupledAgg()))
      val ro = IO(Debug.readAlways(new DecoupledAgg()))
      // Input probe
      // val c = IO(Flipped(Debug.producer(new DecoupledAgg())))

      val data = WireInit(a)
      take.define(data)
      prod.define(data)
      cons.define(data)
      ro.define(data)

      val test = prod.materialize

      a := DontCare
      test := DontCare
    }
    // Check FIRRTL.
    val chirrtl = ChiselStage.emitCHIRRTL(new Example)
    matchesAndOmits(chirrtl)(
      "input a : { flip incoming : { flip ready : UInt<1>, valid : UInt<1>, bits : UInt<8>}, outgoing : { flip ready : UInt<1>, valid : UInt<1>, bits : UInt<8>}}",
      "output take : { incoming : { ready : RWProbe<UInt<1>>, valid : RWProbe<UInt<1>>, bits : RWProbe<UInt<8>>}, outgoing : { ready : RWProbe<UInt<1>>, valid : RWProbe<UInt<1>>, bits : RWProbe<UInt<8>>}}",
      "output prod : { incoming : { ready : Probe<UInt<1>>, valid : RWProbe<UInt<1>>, bits : RWProbe<UInt<8>>}, outgoing : { ready : RWProbe<UInt<1>>, valid : Probe<UInt<1>>, bits : Probe<UInt<8>>}}",
      "output cons : { incoming : { ready : RWProbe<UInt<1>>, valid : Probe<UInt<1>>, bits : Probe<UInt<8>>}, outgoing : { ready : Probe<UInt<1>>, valid : RWProbe<UInt<1>>, bits : RWProbe<UInt<8>>}}",
      "output ro : { incoming : { ready : Probe<UInt<1>>, valid : Probe<UInt<1>>, bits : Probe<UInt<8>>}, outgoing : { ready : Probe<UInt<1>>, valid : Probe<UInt<1>>, bits : Probe<UInt<8>>}}"
    )()

    // And compilation through to SV.
    ChiselStage.emitSystemVerilog((new Example))
  }
  "Debug.producer examples" should "work" in {
    class Test extends Module {
      val in = IO(new DecoupledAgg())

      in.outgoing :<= DontCare
      dontTouch(in.outgoing)
      DontCare :>= in.incoming
      dontTouch(in.incoming)
    }
    class Child extends Module {
      val in = IO(new DecoupledAgg())

      val t = Module(new Test)

      val prod = IO(Debug.producer(t.in))
      // Can't do this presently, "soon": https://github.com/llvm/circt/pull/6258
      val w = Wire(new DecoupledAgg())
      w :<>= t.in
      in :<>= w
      prod.define(w)
    }
    class Example extends Module {
      val in = IO(new DecoupledAgg())
      val debug = IO(new DecoupledAgg())

      val c = Module(new Child)
      in :<>= c.in

      withDisable(Disable.Never) {
        debug :<>= c.prod.materialize
      }
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new Example)

    ChiselStage.emitSystemVerilog((new Example))
    matchesAndOmits(chirrtl)(
      // Child.prod
      "output prod : { incoming : { ready : Probe<UInt<1>>, valid : RWProbe<UInt<1>>, bits : RWProbe<UInt<8>>}, outgoing : { ready : RWProbe<UInt<1>>, valid : Probe<UInt<1>>, bits : Probe<UInt<8>>}}",
      // Check the define
      "define prod.outgoing.bits = probe(w.outgoing.bits)",
      "define prod.outgoing.valid = probe(w.outgoing.valid)",
      "define prod.outgoing.ready = rwprobe(w.outgoing.ready)",
      "define prod.incoming.bits = rwprobe(w.incoming.bits)",
      "define prod.incoming.valid = rwprobe(w.incoming.valid)",
      "define prod.incoming.ready = probe(w.incoming.ready)",
      // Check the force/read bits too
    )()
  }
  "Debug on extmodule" should "work" in {
    class Test extends ExtModule {
      val in = IO(Debug.producer(new DecoupledAgg()))
    }
    class Child extends Module {
      val t = Module(new Test)
      val prod = IO(chiselTypeOf(t.in))
      prod :<>= t.in
    }
    class Example extends Module {
      val debug = IO(new DecoupledAgg())

      val c = Module(new Child)
      withDisable(Disable.Never) {
        debug :<>= c.prod.materialize
      }
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new Example, Array("--full-stacktrace"))

    println(pruneSourceLoc(chirrtl))

    println(ChiselStage.emitSystemVerilog((new Example)))
    matchesAndOmits(chirrtl)(
      "output in : { incoming : { ready : Probe<UInt<1>>, valid : RWProbe<UInt<1>>, bits : RWProbe<UInt<8>>}, outgoing : { ready : RWProbe<UInt<1>>, valid : Probe<UInt<1>>, bits : Probe<UInt<8>>}}"
    )()
  }
}
