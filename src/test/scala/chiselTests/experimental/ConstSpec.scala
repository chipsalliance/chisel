package chiselTests.experimental

import chisel3._
import chiselTests.{ChiselFlatSpec, Utils}
import chisel3.experimental.Const
import chisel3.experimental.BundleLiterals.AddBundleLiteralConstructor
import chisel3.experimental.VecLiterals.AddVecLiteralConstructor
import circt.stage.ChiselStage

class ConstSpec extends ChiselFlatSpec with Utils {

  "Const modifier on a wire or register" should "emit FIRRTL const descriptors" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new Module {
      val foo = Wire(Const(UInt(8.W)))
      val bar = Reg(Const(SInt(4.W)))
    })
    chirrtl should include("wire foo : const UInt<8>")
    chirrtl should include("reg bar : const SInt<4>")
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

  "Bundles initialized with literals" should "emit FIRRTL const descriptors" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new Module {
      class FooBundle extends Bundle {
        val bar = Const(Bool())
      }
      val r = RegInit((new FooBundle).Lit(_.bar -> false.B))
    })
    chirrtl should include("wire _r_WIRE : const { bar : const UInt<1>}")
  }

  "Vectors initialized with literals" should "emit FIRRTL const descriptors" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new Module {
      val r = RegInit(Vec(4, UInt(8.W)).Lit(0 -> 0xab.U(8.W), 2 -> 0xef.U(8.W), 3 -> 0xff.U(8.W)))
    })
    chirrtl should include("wire _r_WIRE : const UInt<8>[4]")
  }

}
