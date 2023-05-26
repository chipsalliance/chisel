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
    exc.getMessage should be("Cannot create register with constant value.")
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

  "Memories of Const type" should "fail" in {
    val exc = intercept[chisel3.ChiselException] {
      ChiselStage.emitCHIRRTL(
        new Module {
          val mem = SyncReadMem(1024, Const(Vec(4, UInt(32.W))))
        },
        Array("--throw-on-first-error")
      )
    }
    exc.getMessage should be("Mem type cannot be const.")
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
    exc.getMessage should be("Cannot create Const of a Probe.")
  }

}
