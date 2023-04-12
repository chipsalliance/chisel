package chiselTests

import chisel3._
import chiselTests.{ChiselFlatSpec, Utils}
import chisel3.experimental.BundleLiterals.AddBundleLiteralConstructor
import chisel3.experimental.VecLiterals.AddVecLiteralConstructor
import circt.stage.ChiselStage

class ConstSpec extends ChiselFlatSpec with Utils {

  "Const modifier on a wire" should "emit FIRRTL const descriptors" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new Module {
      val foo = Wire(Const(UInt(8.W)))
    })
    chirrtl should include("wire foo : const UInt<8>")
  }

  "Const modifier on a register" should "fail" in {
    val err = intercept[java.lang.IllegalArgumentException] {
      ChiselStage.emitCHIRRTL(new Module {
        val foo = Reg(Const(SInt(4.W)))
      })
    }
    err.getMessage should be("requirement failed: Cannot create register with constant value.")
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

  // TODO mems

}
