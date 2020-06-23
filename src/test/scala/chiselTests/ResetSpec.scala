// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.stage.ChiselStage
import chisel3.util.{Counter, Queue}
import chisel3.testers.BasicTester

class ResetAgnosticModule extends RawModule {
  val clk = IO(Input(Clock()))
  val rst = IO(Input(Reset()))
  val out = IO(Output(UInt(8.W)))

  val reg = withClockAndReset(clk, rst)(RegInit(0.U(8.W)))
  reg := reg + 1.U
  out := reg
}

class AbstractResetDontCareModule extends RawModule {
  import chisel3.util.Valid
  val monoPort = IO(Output(Reset()))
  monoPort := DontCare
  val monoWire = Wire(Reset())
  monoWire := DontCare
  val monoAggPort = IO(Output(Valid(Reset())))
  monoAggPort := DontCare
  val monoAggWire = Wire(Valid(Reset()))
  monoAggWire := DontCare

  // Can't bulk connect to Wire so only ports here
  val bulkPort = IO(Output(Reset()))
  bulkPort <> DontCare
  val bulkAggPort = IO(Output(Valid(Reset())))
  bulkAggPort <> DontCare
}


class ResetSpec extends ChiselFlatSpec with Utils {

  behavior of "Reset"

  it should "be able to be connected to DontCare" in {
    ChiselStage.elaborate(new AbstractResetDontCareModule)
  }

  it should "allow writing modules that are reset agnostic" in {
    val sync = compile(new Module {
      val io = IO(new Bundle {
        val out = Output(UInt(8.W))
      })
      val inst = Module(new ResetAgnosticModule)
      inst.clk := clock
      inst.rst := reset
      assert(inst.rst.isInstanceOf[chisel3.ResetType])
      io.out := inst.out
    })
    sync should include ("always @(posedge clk)")

    val async = compile(new Module {
      val io = IO(new Bundle {
        val out = Output(UInt(8.W))
      })
      val inst = Module(new ResetAgnosticModule)
      inst.clk := clock
      inst.rst := reset.asTypeOf(AsyncReset())
      assert(inst.rst.isInstanceOf[chisel3.ResetType])
      io.out := inst.out
    })
    async should include ("always @(posedge clk or posedge rst)")
  }

  behavior of "Users"

  they should "be able to force implicit reset to be synchronous" in {
    val fir = (new ChiselStage).emitChirrtl(new MultiIOModule with RequireSyncReset {
      reset shouldBe a [Bool]
    })
    fir should include ("input reset : UInt<1>")
  }

  they should "be able to force implicit reset to be asynchronous" in {
    val fir = (new ChiselStage).emitChirrtl(new MultiIOModule with RequireAsyncReset {
      reset shouldBe an [AsyncReset]
    })
    fir should include ("input reset : AsyncReset")
  }

  "Chisel" should "error if sync and async modules are nested" in {
    a [ChiselException] should be thrownBy extractCause[ChiselException] {
      ChiselStage.elaborate(new MultiIOModule with RequireAsyncReset {
        val mod = Module(new MultiIOModule with RequireSyncReset)
      })
    }
  }
}
