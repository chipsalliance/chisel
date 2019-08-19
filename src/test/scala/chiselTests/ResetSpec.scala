// See LICENSE for license details.

package chiselTests

import chisel3._
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


class ResetSpec extends ChiselFlatSpec {

  behavior of "Reset"

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
}
