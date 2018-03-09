package chisel3.tests

import org.scalatest._

import chisel3._
import chisel3.tester._

class BasicTest extends FlatSpec with ChiselScalatestTester {
  behavior of "Testers2"

  private val backendNames = Array[String] ("firrtl", "verilator")
  for (backendName <- backendNames) {
    val options = new TesterOptionsManager {
      testerOptions = TesterOptions(backendName = backendName)
    }

    it should s"test static circuits (with $backendName)" in {
      test(new Module {
        val io = IO(new Bundle {
          val out = Output(UInt(8.W))
        })
        io.out := 42.U
      }, options) { c =>
        c.io.out.expect(42.U)
      }
    }
    it should s"test inputless sequential circuits (with $backendName)" in {
      test(new Module {
        val io = IO(new Bundle {
          val out = Output(UInt(8.W))
        })
        val counter = RegInit(UInt(8.W), 0.U)
        counter := counter + 1.U
        io.out := counter
      }, options) { c =>
        c.io.out.expect(0.U)
        c.clock.step()
        c.io.out.expect(1.U)
        c.clock.step()
        c.io.out.expect(2.U)
        c.clock.step()
        c.io.out.expect(3.U)
      }
    }

    it should s"test combinational circuits (with $backendName)" in {
      test(new Module {
        val io = IO(new Bundle {
          val in = Input(UInt(8.W))
          val out = Output(UInt(8.W))
        })
        io.out := io.in
      }, options) { c =>
        c.io.in.poke(0.U)
        c.io.out.expect(0.U)
        c.io.in.poke(42.U)
        c.io.out.expect(42.U)
      }
    }

    it should s"test sequential circuits (with $backendName)" in {
      test(new Module {
        val io = IO(new Bundle {
          val in = Input(UInt(8.W))
          val out = Output(UInt(8.W))
        })
        io.out := RegNext(io.in, 0.U)
      }, options) { c =>
        c.io.in.poke(0.U)
        c.clock.step()
        c.io.out.expect(0.U)
        c.io.in.poke(42.U)
        c.clock.step()
        c.io.out.expect(42.U)
      }
    }
  }
}
