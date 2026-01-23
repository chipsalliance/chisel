package chiselTests

import chisel3._
import chisel3.util.Decoupled
import circt.stage.ChiselStage
import org.scalatest.matchers.should.Matchers
import org.scalatest.propspec.AnyPropSpec
import scala.annotation.nowarn

class BulkConnectSpec extends AnyPropSpec with Matchers {
  property("Chisel connects should emit FIRRTL bulk connects when possible") {
    val chirrtl = ChiselStage.emitCHIRRTL(new Module {
      val io = IO(new Bundle {
        val inMono = Input(Vec(4, UInt(8.W)))
        val outMono = Output(Vec(4, UInt(8.W)))
        val inBi = Input(Vec(4, UInt(8.W)))
        val outBi = Output(Vec(4, UInt(8.W)))
      })
      io.outMono := io.inMono
      io.outBi <> io.inBi
    })
    chirrtl should include("connect io.outMono, io.inMono")
    chirrtl should include("connect io.outBi, io.inBi")
  }

  property("Chisel connects should not emit FIRRTL bulk connects between differing FIRRTL types") {
    val chirrtl = ChiselStage.emitCHIRRTL(new Module {
      val in = IO(Flipped(new Bundle {
        val foo = Flipped(new Bundle {
          val bar = Input(UInt(8.W))
        })
      }))
      val out = IO(Output(new Bundle {
        val foo = new Bundle {
          val bar = UInt(8.W)
        }
      }))
      // Both of these connections are legal in Chisel, but in and out do not have the same type
      out := in
      out <> in
    })
    // out <- in is illegal FIRRTL
    exactly(2, chirrtl.split('\n')) should include("connect out.foo.bar, in.foo.bar")
    chirrtl shouldNot include("connect out, in")
    chirrtl shouldNot include("out <- in")
  }

  property("Chisel connects should not emit a FIRRTL bulk connect for a bidirectional MonoConnect") {
    val chirrtl = ChiselStage.emitCHIRRTL(new Module {
      val enq = IO(Flipped(Decoupled(UInt(8.W))))
      val deq = IO(Decoupled(UInt(8.W)))

      // Implicitly create a MonoConnect from enq to a wire
      // enq is a Decoupled and so has input/output signals
      // We should not bulk connect in this case
      val wire = WireDefault(enq)
      dontTouch(wire)
      deq <> enq
    })

    chirrtl shouldNot include("connect wire, enq")
    chirrtl should include("connect wire.bits, enq.bits")
    chirrtl should include("connect wire.valid, enq.valid")
    chirrtl should include("connect wire.ready, enq.ready")
    chirrtl should include("connect deq, enq")
  }

  property("Chisel connects should not emit a FIRRTL bulk connect for BlackBox IO Bundles") {
    class MyBundle extends Bundle {
      val O: Bool = Output(Bool())
      val I: Bool = Input(Bool())
    }

    val chirrtl = ChiselStage.emitCHIRRTL(new Module {
      val io: MyBundle = IO(Flipped(new MyBundle))

      @nowarn("cat=deprecation")
      val bb = Module(new BlackBox {
        val io: MyBundle = IO(Flipped(new MyBundle))
      })

      io <> bb.io
    })
    // There won't be a bb.io Bundle in FIRRTL, so connections have to be done element-wise
    chirrtl should include("connect bb.O, io.O")
    chirrtl should include("connect io.I, bb.I")
  }

  property("MonoConnect should bulk connect undirectioned internal wires") {
    val chirrtl = ChiselStage.emitCHIRRTL(new Module {
      val io = IO(new Bundle {})
      val w1 = Wire(Vec(2, UInt(8.W)))
      val w2 = Wire(Vec(2, UInt(8.W)))
      w2 := w1
    })
    chirrtl should include("connect w2, w1")
  }

  property("Chisel connects should not emit a FIRRTL bulk connect for single target but non-identity views") {
    import chisel3.experimental.dataview._
    type ReversedVec[T <: Data] = Vec[T]
    implicit def reversedVecView[T <: Data]: DataView[Vec[T], ReversedVec[T]] =
      DataView.mapping[Vec[T], ReversedVec[T]](v => v.cloneType, { case (a, b) => a.reverse.zip(b) })
    val chirrtl = ChiselStage.emitCHIRRTL(new Module {
      val in0, in1 = IO(Input(Vec(2, UInt(8.W))))
      val out0, out1 = IO(Output(Vec(2, UInt(8.W))))

      out0 := in0.viewAs[ReversedVec[UInt]]
      out1 <> in1.viewAs[ReversedVec[UInt]]
    })
    chirrtl shouldNot include("connect out0, in0")
    chirrtl should include("connect out0[0], in0[1]")
    chirrtl should include("connect out0[1], in0[0]")
    chirrtl shouldNot include("connect out1, in1")
    chirrtl should include("connect out1[0], in1[1]")
    chirrtl should include("connect out1[1], in1[0]")
  }

  property("Chisel should emit FIRRTL bulk connect for \"input\" wires") {
    class MyBundle extends Bundle {
      val foo = Input(UInt(8.W))
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new Module {
      val w1, w2 = Wire(new MyBundle)
      w2 <> w1
    })
    chirrtl should include("connect w2, w1")
  }
}
