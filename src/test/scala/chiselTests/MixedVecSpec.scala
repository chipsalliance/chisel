// SPDX-License-Identifier: Apache-2.0

package chiselTests

import circt.stage.ChiselStage
import chisel3._
import chisel3.util.MixedVec

// Most tests are in MixedVecIntegrationSpec
class MixedVecSpec extends ChiselPropSpec with Utils {

  property("MixedVecs should not be able to take hardware types") {
    a[ExpectedChiselTypeException] should be thrownBy extractCause[ExpectedChiselTypeException] {
      ChiselStage.emitCHIRRTL(new Module {
        val io = IO(new Bundle {})
        val hw = Wire(MixedVec(Seq(UInt(8.W), Bool())))
        val illegal = MixedVec(hw)
      })
    }
    a[ExpectedChiselTypeException] should be thrownBy extractCause[ExpectedChiselTypeException] {
      ChiselStage.emitCHIRRTL(new Module {
        val io = IO(new Bundle {})
        val hw = Reg(MixedVec(Seq(UInt(8.W), Bool())))
        val illegal = MixedVec(hw)
      })
    }
    a[ExpectedChiselTypeException] should be thrownBy extractCause[ExpectedChiselTypeException] {
      ChiselStage.emitCHIRRTL(new Module {
        val io = IO(new Bundle {
          val v = Input(MixedVec(Seq(UInt(8.W), Bool())))
        })
        val illegal = MixedVec(io.v)
      })
    }
  }

  property("Connecting a MixedVec and something of different size should report a ChiselException") {
    an[IllegalArgumentException] should be thrownBy extractCause[IllegalArgumentException] {
      ChiselStage.emitCHIRRTL(new Module {
        val io = IO(new Bundle {
          val out = Output(MixedVec(Seq(UInt(8.W), Bool())))
        })
        val seq = Seq.fill(5)(0.U)
        io.out := seq
      })
    }
    an[IllegalArgumentException] should be thrownBy extractCause[IllegalArgumentException] {
      ChiselStage.emitCHIRRTL(new Module {
        val io = IO(new Bundle {
          val out = Output(MixedVec(Seq(UInt(8.W), Bool())))
        })
        val seq = VecInit(Seq(100.U(8.W)))
        io.out := seq
      })
    }
  }

  property("MixedVec connections should emit FIRRTL bulk connects when possible") {
    val chirrtl = ChiselStage.emitCHIRRTL(new Module {
      val io = IO(new Bundle {
        val inMono = Input(MixedVec(Seq(UInt(8.W), UInt(16.W), UInt(4.W), UInt(7.W))))
        val outMono = Output(MixedVec(Seq(UInt(8.W), UInt(16.W), UInt(4.W), UInt(7.W))))
        val inBi = Input(MixedVec(Seq(UInt(8.W), UInt(16.W), UInt(4.W), UInt(7.W))))
        val outBi = Output(MixedVec(Seq(UInt(8.W), UInt(16.W), UInt(4.W), UInt(7.W))))
      })
      // Explicit upcast avoids weird issue where Scala 2.12 overloading resolution calls version of := accepting Seq[T] instead of normal Data version
      io.outMono := (io.inMono: Data)
      io.outBi <> io.inBi
    })
    chirrtl should include("connect io.outMono, io.inMono")
    chirrtl should include("connect io.outBi, io.inBi")
  }

  property("MixedVec should be a Seq") {
    // Compile Only Check
    class Foo extends Module {
      val io = IO(new Bundle {
        val inMono = Input(MixedVec(Seq(UInt(8.W), UInt(16.W), UInt(4.W), UInt(7.W))))
      })
      val foo: Seq[UInt] = io.inMono
    }
  }
}
