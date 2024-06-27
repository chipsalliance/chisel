package chiselTests.simulator

import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.must.Matchers

import chisel3._
import chisel3.util.Valid
import chisel3.simulator.EphemeralSimulator._

class GCD2 extends Module {
  val io = IO(new Bundle {
    val in = Flipped(Valid(Vec(2, UInt(32.W))))
    val result = Valid(UInt(32.W))
  })
  val x = Reg(UInt(32.W))
  val y = Reg(UInt(32.W))
  when(x > y) { x := x -% y }.otherwise { y := y -% x }
  when(io.in.valid) { x := io.in.bits(0); y := io.in.bits(1) }
  io.result.bits := x
  io.result.valid := y === 0.U
}

class EphemeralSimulatorSpec extends AnyFunSpec with Matchers {
  describe("EphemeralSimulator") {
    it("runs GCD correctly") {
      simulate(new GCD()) { gcd =>
        gcd.io.a.poke(24.U)
        gcd.io.b.poke(36.U)
        gcd.io.loadValues.poke(1.B)
        gcd.clock.step()
        gcd.io.loadValues.poke(0.B)
        gcd.clock.stepUntil(sentinelPort = gcd.io.resultIsValid, sentinelValue = 1, maxCycles = 10)
        gcd.io.resultIsValid.expect(true.B)
        gcd.io.result.expect(12)
      }
    }

    it("runs GCD2 correctly") {
      simulate(new GCD2()) { gcd =>
        gcd.io.in.bits(0).poke(24.U)
        gcd.io.in.bits(1).poke(36.U)
        gcd.io.in.valid.poke(1.B)
        gcd.clock.step()
        gcd.io.in.valid.poke(0.B)
        gcd.clock.stepUntil(sentinelPort = gcd.io.result.valid, sentinelValue = 1, maxCycles = 10)
        gcd.io.result.valid.expect(true.B)
        gcd.io.result.bits.expect(12)
      }
    }
  }
}
