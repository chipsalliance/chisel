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

  "Debug examples" should "work" in {
    class DecoupledAgg extends Bundle {
      val incoming = Flipped(DecoupledIO(UInt(8.W)))
      val outgoing = DecoupledIO(UInt(8.W))
    }
    class InputOutputTest extends Bundle {
      val incoming = Input(DecoupledIO(UInt(8.W)))
      val outgoing = Output(DecoupledIO(UInt(8.W)))
    }
    class Example extends Module {
      val a = IO(new DecoupledAgg())
      // { output a: {
      //     flipped incoming: { flipped ready: UInt<1>, valid: UInt<1> },
      //             outgoing: { flipped ready: UInt<1>, valid: UInt<1> },
      // }

      val take = IO(Debug.takeOver(new DecoupledAgg()))
      val prod = IO(Debug.producer(new DecoupledAgg()))
      val cons = IO(Debug.consumer(new DecoupledAgg()))
      val ro = IO(Debug.readAlways(new DecoupledAgg()))
      // val test = Debug.producer(a)
      // val c = IO(Flipped(Debug.producer(new DecoupledAgg())))
      // val b = IO(Debug.producer(IO(new DecoupledAgg())))

      // val b = IO(new Debug(new DecoupledIO(UInt(8.W)), ProducerKind))

//      println(s"isFullyAligned(DecoupledAgg): ${DataMirror.isFullyAligned(new DecoupledAgg())}")
//      println(s"isFullyAligned(DecoupledAgg.incoming): ${DataMirror.isFullyAligned(new DecoupledAgg().incoming)}")
//      println(s"isFullyAligned(DecoupledAgg.outgoing): ${DataMirror.isFullyAligned(new DecoupledAgg().outgoing)}")
//      println(s"isFullyAligned(Output(DecoupledAgg)): ${DataMirror.isFullyAligned(Output(new DecoupledAgg()))}")
//      println(s"isFullyAligned(Input(DecoupledAgg)): ${DataMirror.isFullyAligned(Input(new DecoupledAgg()))}")
//
//      println(s"isFullyAligned(InputOutputTest.incoming): ${DataMirror.isFullyAligned(new InputOutputTest().incoming)}")
//      println(s"isFullyAligned(InputOutputTest.outgoing): ${DataMirror.isFullyAligned(new InputOutputTest().outgoing)}")

      val test = prod.materialize
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new Example)

    println(chirrtl)

    matchesAndOmits(chirrtl)(
      "output a : { flip incoming : { flip ready : UInt<1>, valid : UInt<1>, bits : UInt<8>}, outgoing : { flip ready : UInt<1>, valid : UInt<1>, bits : UInt<8>}}",
      "output take : { incoming : { ready : RWProbe<UInt<1>>, valid : RWProbe<UInt<1>>, bits : RWProbe<UInt<8>>}, outgoing : { ready : RWProbe<UInt<1>>, valid : RWProbe<UInt<1>>, bits : RWProbe<UInt<8>>}}",
      "output prod : { incoming : { ready : Probe<UInt<1>>, valid : RWProbe<UInt<1>>, bits : RWProbe<UInt<8>>}, outgoing : { ready : RWProbe<UInt<1>>, valid : Probe<UInt<1>>, bits : Probe<UInt<8>>}}",
      "output cons : { incoming : { ready : RWProbe<UInt<1>>, valid : Probe<UInt<1>>, bits : Probe<UInt<8>>}, outgoing : { ready : Probe<UInt<1>>, valid : RWProbe<UInt<1>>, bits : RWProbe<UInt<8>>}}",
      "output ro : { incoming : { ready : Probe<UInt<1>>, valid : Probe<UInt<1>>, bits : Probe<UInt<8>>}, outgoing : { ready : Probe<UInt<1>>, valid : Probe<UInt<1>>, bits : Probe<UInt<8>>}}"
    )()
  }
  "Debug.producer examples" should "work" in {
    class DecoupledAgg extends Bundle {
      val incoming = Flipped(DecoupledIO(UInt(8.W)))
      val outgoing = DecoupledIO(UInt(8.W))
    }
    class Test extends Module {
      val in = IO(new DecoupledAgg())

      // in.outgoing :<>= in.incoming
      in.outgoing :<= DontCare
      dontTouch(in.outgoing)
      DontCare :>= in.incoming
      dontTouch(in.incoming)
    }
    class Child extends Module {
      val in = IO(new DecoupledAgg())

      val t = Module(new Test)
      // in :<>= t.in

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
      // c.in :<>= in
      in :<>= c.in

      withDisable(Disable.Never) {
        debug :<>= c.prod.materialize

        // TODO: Test wire gets name, which it seems to.
        // val m = c.prod.materialize
        // debug :<>= m
      }
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new Example)

    println(pruneSourceLoc(chirrtl))

    println(ChiselStage.emitSystemVerilog((new Example)))
    // matchesAndOmits(chirrtl)(
    //   "output a : { flip incoming : { flip ready : UInt<1>, valid : UInt<1>, bits : UInt<8>}, outgoing : { flip ready : UInt<1>, valid : UInt<1>, bits : UInt<8>}}",
    //   "output take : { incoming : { ready : RWProbe<UInt<1>>, valid : RWProbe<UInt<1>>, bits : RWProbe<UInt<8>>}, outgoing : { ready : RWProbe<UInt<1>>, valid : RWProbe<UInt<1>>, bits : RWProbe<UInt<8>>}}",
    //   "output prod : { incoming : { ready : Probe<UInt<1>>, valid : RWProbe<UInt<1>>, bits : RWProbe<UInt<8>>}, outgoing : { ready : RWProbe<UInt<1>>, valid : Probe<UInt<1>>, bits : Probe<UInt<8>>}}",
    //   "output cons : { incoming : { ready : RWProbe<UInt<1>>, valid : Probe<UInt<1>>, bits : Probe<UInt<8>>}, outgoing : { ready : Probe<UInt<1>>, valid : RWProbe<UInt<1>>, bits : RWProbe<UInt<8>>}}",
    //   "output ro : { incoming : { ready : Probe<UInt<1>>, valid : Probe<UInt<1>>, bits : Probe<UInt<8>>}, outgoing : { ready : Probe<UInt<1>>, valid : Probe<UInt<1>>, bits : Probe<UInt<8>>}}"
    // )()
  }
  "Debug on extmodule" should "work" in {
    class DecoupledAgg extends Bundle {
      val incoming = Flipped(DecoupledIO(UInt(8.W)))
      val outgoing = DecoupledIO(UInt(8.W))
    }
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
      // c.in :<>= in
      withDisable(Disable.Never) {
        debug :<>= c.prod.materialize

        // TODO: Test wire gets name, which it seems to.
        // val m = c.prod.materialize
        // debug :<>= m
      }
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new Example, Array("--full-stacktrace"))

    println(pruneSourceLoc(chirrtl))

    println(ChiselStage.emitSystemVerilog((new Example)))
    // matchesAndOmits(chirrtl)(
    //   "output a : { flip incoming : { flip ready : UInt<1>, valid : UInt<1>, bits : UInt<8>}, outgoing : { flip ready : UInt<1>, valid : UInt<1>, bits : UInt<8>}}",
    //   "output take : { incoming : { ready : RWProbe<UInt<1>>, valid : RWProbe<UInt<1>>, bits : RWProbe<UInt<8>>}, outgoing : { ready : RWProbe<UInt<1>>, valid : RWProbe<UInt<1>>, bits : RWProbe<UInt<8>>}}",
    //   "output prod : { incoming : { ready : Probe<UInt<1>>, valid : RWProbe<UInt<1>>, bits : RWProbe<UInt<8>>}, outgoing : { ready : RWProbe<UInt<1>>, valid : Probe<UInt<1>>, bits : Probe<UInt<8>>}}",
    //   "output cons : { incoming : { ready : RWProbe<UInt<1>>, valid : Probe<UInt<1>>, bits : Probe<UInt<8>>}, outgoing : { ready : Probe<UInt<1>>, valid : RWProbe<UInt<1>>, bits : RWProbe<UInt<8>>}}",
    //   "output ro : { incoming : { ready : Probe<UInt<1>>, valid : Probe<UInt<1>>, bits : Probe<UInt<8>>}, outgoing : { ready : Probe<UInt<1>>, valid : Probe<UInt<1>>, bits : Probe<UInt<8>>}}"
    // )()
  }
}
