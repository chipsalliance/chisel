package chisel3.tests

import org.scalatest._

import chisel3._
import chisel3.tester._

class CombinationalPathTest extends FlatSpec with ChiselScalatestTester {
  behavior of "Testers2"

  it should "detect combinationally-dependent operations across threads" in {
    assertThrows[ThreadOrderDependentException] {
      test(new Module {
        val io = IO(new Bundle {
          val in = Input(Bool())
          val out = Output(Bool())
        })
        io.out := io.in
      }) { c =>
        c.io.in.poke(false.B)
        fork {
          c.io.out.expect(false.B)
        }
      }
    }
  }

  it should "detect combinationally-dependent operations through internal modules" in {
    assertThrows[ThreadOrderDependentException] {
      test(new Module {
        val io = IO(new Bundle {
          val in = Input(Bool())
          val out = Output(Bool())
        })

        val innerModule = Module(new Module {
          val io = IO(new Bundle {
            val in = Input(Bool())
            val out = Output(Bool())
          })
          io.out := io.in
        })

        innerModule.io.in := io.in
        io.out := innerModule.io.out
      }) { c =>
        c.io.in.poke(false.B)
        fork {
          c.io.out.expect(false.B)
        }
      }
    }
  }

  it should "detect combinational paths across operations" in {
    assertThrows[ThreadOrderDependentException] {
      test(new Module {
        val io = IO(new Bundle {
          val in1 = Input(Bool())
          val in2 = Input(Bool())
          val out = Output(Bool())
        })
        io.out := io.in1 && io.in2
      }) { c =>
        c.io.in1.poke(false.B)
        fork {
          c.io.out.expect(false.B)
        }
      }
    }
  }
}
