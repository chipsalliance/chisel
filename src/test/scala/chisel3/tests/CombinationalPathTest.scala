package chisel3.tests

import org.scalatest._

import chisel3._
import chisel3.tester._

class CombinationalPathTest extends FlatSpec with ChiselScalatestTester {
  behavior of "Testers2"

  it should "detect combinational loops" in {
    // TODO this test should fail
    test(new Module {
      val io = IO(new Bundle {
        val in1 = Input(UInt(8.W))
        val in2 = Input(UInt(8.W))
        val out1 = Output(UInt(8.W))
        val out2 = Output(UInt(8.W))
      })
      io.out1 := io.in1 + io.in2

      val innerModule = Module(new Module {
        val io = IO(new Bundle {
          val in = Input(UInt(8.W))
          val out = Output(UInt(8.W))
        })
        io.out := io.in
      })
      innerModule.io.in := io.in1
      io.out2 := innerModule.io.out
    }) { c =>
      fork {
        c.io.in1.poke(1.U)
      } .fork {
        c.io.in2.poke(2.U)
      }
    }
  }
}
