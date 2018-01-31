package chisel3.tests

import org.scalatest._

import chisel3._
import chisel3.testers2._

class BasicTest extends FlatSpec with ChiselScalatestTester {
  behavior of "Testers2"

  it should "test static circuits" in {
    test(Firrterpreter.start(new Module {
      val io = IO(new Bundle {
        val out = Output(UInt(8.W))
      })
      io.out := 42.U
    })) { c =>
      c.io.out.check(42.U)
    }
  }

  it should "test inputless sequential circuits" in {
    test(Firrterpreter.start(new Module {
      val io = IO(new Bundle {
        val out = Output(UInt(8.W))
      })
      val counter = RegInit(UInt(8.W), 0.U)
      counter := counter + 1.U
      io.out := counter
    })) { c =>
      c.io.out.check(0.U)
      c.clock.step()
      c.io.out.check(1.U)
      c.clock.step()
      c.io.out.check(2.U)
      c.clock.step()
      c.io.out.check(3.U)
    }
  }

  it should "test combinational circuits" in {
    test(Firrterpreter.start(new Module {
      val io = IO(new Bundle {
        val in = Input(UInt(8.W))
        val out = Output(UInt(8.W))
      })
      io.out := io.in
    })) { c =>
      c.io.in.poke(0.U)
      c.io.out.check(0.U)
      c.io.in.poke(42.U)
      c.io.out.check(42.U)
    }
  }

  it should "test sequential circuits" in {
    test(Firrterpreter.start(new Module {
      val io = IO(new Bundle {
        val in = Input(UInt(8.W))
        val out = Output(UInt(8.W))
      })
      io.out := RegNext(io.in, 0.U)
    })) { c =>
      c.io.in.poke(0.U)
      c.clock.step()
      c.io.out.check(0.U)
      c.io.in.poke(42.U)
      c.clock.step()
      c.io.out.check(42.U)
    }
  }
}
