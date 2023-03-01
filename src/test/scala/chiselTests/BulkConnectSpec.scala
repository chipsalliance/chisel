package chiselTests

import chisel3._
import chisel3.util.Decoupled
import circt.stage.ChiselStage
import chisel3.testers.BasicTester

import scala.annotation.nowarn

class BulkConnectSpec extends ChiselPropSpec {
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
    chirrtl should include("io.outMono <= io.inMono")
    chirrtl should include("io.outBi <= io.inBi")
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
    exactly(2, chirrtl.split('\n')) should include("out.foo.bar <= in.foo.bar")
    chirrtl shouldNot include("out <= in")
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

    chirrtl shouldNot include("wire <= enq")
    chirrtl should include("wire.bits <= enq.bits")
    chirrtl should include("wire.valid <= enq.valid")
    chirrtl should include("wire.ready <= enq.ready")
    chirrtl should include("deq <= enq")
  }

  property("Chisel connects should not emit a FIRRTL bulk connect for BlackBox IO Bundles") {
    class MyBundle extends Bundle {
      val O: Bool = Output(Bool())
      val I: Bool = Input(Bool())
    }

    val chirrtl = ChiselStage.emitCHIRRTL(new Module {
      val io: MyBundle = IO(Flipped(new MyBundle))

      val bb = Module(new BlackBox {
        val io: MyBundle = IO(Flipped(new MyBundle))
      })

      io <> bb.io
    })
    // There won't be a bb.io Bundle in FIRRTL, so connections have to be done element-wise
    chirrtl should include("bb.O <= io.O")
    chirrtl should include("io.I <= bb.I")
  }

  property("MonoConnect should bulk connect undirectioned internal wires") {
    val chirrtl = ChiselStage.emitCHIRRTL(new Module {
      val io = IO(new Bundle {})
      val w1 = Wire(Vec(2, UInt(8.W)))
      val w2 = Wire(Vec(2, UInt(8.W)))
      w2 := w1
    })
    chirrtl should include("w2 <= w1")
  }
}
