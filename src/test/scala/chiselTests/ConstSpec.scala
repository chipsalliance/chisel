package chiselTests

import chisel3._
import chisel3.probe.Probe
import chisel3.experimental.BundleLiterals.AddBundleLiteralConstructor
import chisel3.experimental.VecLiterals.AddVecLiteralConstructor
import chiselTests.{ChiselFlatSpec, Utils}
import circt.stage.ChiselStage

class ConstSpec extends ChiselFlatSpec with Utils {

  "Const modifier on a wire" should "emit FIRRTL const descriptors" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new Module {
      val foo = Wire(Const(UInt(8.W)))
    })
    chirrtl should include("wire foo : const UInt<8>")
  }

  "Const modifier on a register" should "fail" in {
    val exc = intercept[chisel3.ChiselException] {
      ChiselStage.emitCHIRRTL(
        new Module {
          val foo = Reg(Const(SInt(4.W)))
        },
        Array("--throw-on-first-error")
      )
    }
    exc.getMessage should include("Cannot create register with constant value.")
  }

  "Const modifier on I/O" should "emit FIRRTL const descriptors" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new Module {
      val io = IO(new Bundle {
        val in = Input(Const(UInt(8.W)))
        val out = Output(Const(UInt(8.W)))
      })
    })
    chirrtl should include("output io : { flip in : const UInt<8>, out : const UInt<8>}")
  }

  "Const modifier on bundles and vectors" should "emit FIRRTL const descriptors" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new Module {
      val io = IO(Const(new Bundle {
        val in = Input(Const(Vec(5, AsyncReset())))
        val out = Output(Const(Bool()))
      }))
    })
    chirrtl should include("output io : const { flip in : const AsyncReset[5], out : const UInt<1>}")
  }

  "Const modifier on vectors of const elements" should "emit a single FIRRTL const descriptor" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new Module {
      val foo = Wire(Const(Vec(3, Const(UInt(8.W)))))
      val bar = Wire(Const(Vec(3, Const(Vec(2, Const(UInt(8.W)))))))
      val baz = Wire(Const(Vec(3, Vec(2, Const(UInt(8.W))))))
    })
    chirrtl should include("wire foo : const UInt<8>[3]")
    chirrtl should include("wire bar : const UInt<8>[2][3]")
    chirrtl should include("wire baz : const UInt<8>[2][3]")
  }

  "Memories of Const type" should "fail" in {
    val exc = intercept[chisel3.ChiselException] {
      ChiselStage.emitCHIRRTL(
        new Module {
          val mem = SyncReadMem(1024, Const(Vec(4, UInt(32.W))))
        },
        Array("--throw-on-first-error")
      )
    }
    exc.getMessage should include("Mem type cannot be const.")
  }

  "Const of Probe" should "fail" in {
    val exc = intercept[chisel3.ChiselException] {
      ChiselStage.emitCHIRRTL(
        new Module {
          val p = Const(Probe(Bool()))
        },
        Array("--throw-on-first-error")
      )
    }
    exc.getMessage should include("Cannot create Const of a Probe.")
  }

  class FooBundle extends Bundle {
    val a = UInt(4.W)
    val b = Bool()
  }

  "Const to const connections" should "emit a direct passthrough connection" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new Module {
      val in = IO(Input(Const(new FooBundle)))
      val out = IO(Output(Const(new FooBundle)))
      out := in
    })
    chirrtl should include("connect out, in")
  }

  "Const to non-const connections" should "emit a direct passthrough connection" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new Module {
      val in = IO(Input(Const(new FooBundle)))
      val out = IO(Output(new FooBundle))
      out := in
    })
    chirrtl should include("connect out, in")
  }

  "Const to nested const connections" should "emit a direct passthrough connection" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new Module {
      val in = IO(Input(Const(new Bundle {
        val foo = new FooBundle
      })))
      val out = IO(Output(Const(new Bundle {
        val foo = Const(new FooBundle)
      })))
      out := in
    })
    chirrtl should include("connect out, in")
  }

  class BidirectionalBundle extends Bundle {
    val a = UInt(4.W)
    val b = Flipped(Bool())
  }

  "Const to const bidirection connections using ':<>='" should "emit a direct passthrough connection" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new Module {
      val in = IO(Const(Flipped(new BidirectionalBundle)))
      val out = IO(Const(new BidirectionalBundle))
      out :<>= in
    })
    chirrtl should include("connect out, in")
  }

  "Const to const bidirection connections using '<>'" should "emit a direct passthrough connection" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new Module {
      val in = IO(Const(Flipped(new BidirectionalBundle)))
      val out = IO(Const(new BidirectionalBundle))
      out <> in
    })
    chirrtl should include("connect out, in")
  }

  "Const to const coercing mono connections using ':#='" should "emit elementwise connections" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new Module {
      val in = IO(Const(Flipped(new BidirectionalBundle)))
      val out = IO(Output(Const(new BidirectionalBundle)))
      out :#= in
    })
    chirrtl should include("connect out.b, in.b")
    chirrtl should include("connect out.a, in.a")
  }

}
