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

      val io = WireInit(a)
      take.define(io)
      prod.define(io)
      cons.define(io)
      ro.define(io)

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
  "Debug.producer example start-to-finish" should "work" in {
    class Leaf extends Module {
      val io = IO(new DecoupledAgg())

      io.outgoing :<= DontCare
      dontTouch(io.outgoing)
      DontCare :>= io.incoming
      dontTouch(io.incoming)
    }
    class Mid extends Module {
      val io = IO(new DecoupledAgg())

      val t = Module(new Leaf)

      val prod = IO(Debug.producer(t.io))
      // Can't do this presently, "soon": https://github.com/llvm/circt/pull/6258
      //prod.define(t.io)
      //io:<>= t.io
      val w = Wire(new DecoupledAgg())
      w :<>= t.io
      io :<>= w
      prod.define(w)
    }
    class Example extends Module {
      val io = IO(new DecoupledAgg())
      val debug = IO(new DecoupledAgg())

      val c = Module(new Mid)
      io :<>= c.io

      withDisable(Disable.Never) {
        debug :<>= c.prod.materialize
      }
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new Example)
    matchesAndOmits(chirrtl)(
      // Mid.prod
      "output prod : { incoming : { ready : Probe<UInt<1>>, valid : RWProbe<UInt<1>>, bits : RWProbe<UInt<8>>}, outgoing : { ready : RWProbe<UInt<1>>, valid : Probe<UInt<1>>, bits : Probe<UInt<8>>}}",
      // Check the define (using explicit bounce wire for now)
      "define prod.outgoing.bits = probe(w.outgoing.bits)",
      "define prod.outgoing.valid = probe(w.outgoing.valid)",
      "define prod.outgoing.ready = rwprobe(w.outgoing.ready)",
      "define prod.incoming.bits = rwprobe(w.incoming.bits)",
      "define prod.incoming.valid = rwprobe(w.incoming.valid)",
      "define prod.incoming.ready = probe(w.incoming.ready)",
      // Check the materialize wire and generated read/force's
      "wire _debug_WIRE : { flip incoming : { flip ready : UInt<1>, valid : UInt<1>, bits : UInt<8>}, outgoing : { flip ready : UInt<1>, valid : UInt<1>, bits : UInt<8>}}",
      "connect _debug_WIRE.outgoing.bits, read(c.prod.outgoing.bits)",
      "connect _debug_WIRE.outgoing.valid, read(c.prod.outgoing.valid)",
      "force_initial(c.prod.outgoing.ready, _debug_WIRE.outgoing.ready)",
      "force_initial(c.prod.incoming.bits, _debug_WIRE.incoming.bits)",
      "force_initial(c.prod.incoming.valid, _debug_WIRE.incoming.valid)",
      "connect _debug_WIRE.incoming.ready, read(c.prod.incoming.ready)",
      "connect debug, _debug_WIRE"
    )()

    ChiselStage.emitSystemVerilog((new Example))
  }
  "Debug on extmodule" should "work" in {
    class Leaf extends ExtModule {
      val io = IO(Debug.producer(new DecoupledAgg()))
    }
    class Mid extends Module {
      val t = Module(new Leaf)
      val prod = IO(chiselTypeOf(t.io))
      prod :<>= t.io
    }
    class Example extends Module {
      val debug = IO(new DecoupledAgg())

      val c = Module(new Mid)
      withDisable(Disable.Never) {
        debug :<>= c.prod.materialize
      }
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new Example)
    matchesAndOmits(chirrtl)(
      "output io : { incoming : { ready : Probe<UInt<1>>, valid : RWProbe<UInt<1>>, bits : RWProbe<UInt<8>>}, outgoing : { ready : RWProbe<UInt<1>>, valid : Probe<UInt<1>>, bits : Probe<UInt<8>>}}"
    )()

    ChiselStage.emitSystemVerilog((new Example))
  }

  "Debug.asUInt" should "fail" in {
    val exc = intercept[chisel3.ChiselException] {
      ChiselStage.emitCHIRRTL(new RawModule {
        val u = Debug.producer(Bool()).asUInt
      }, Array("--throw-on-first-error"))
    }
    exc.getMessage should include("Debug does not support .asUInt")
  }
}
