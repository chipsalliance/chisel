package chiselTests.simulator

import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.must.Matchers

import chisel3._
import chisel3.layer.{block, Convention, Layer}
import chisel3.ltl.AssertProperty
import chisel3.simulator.LayerControl
import chisel3.simulator.EphemeralSimulator._

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
    describe("layer control functionality") {
      object A extends Layer(Convention.Bind)
      class Foo extends Module {
        block(A) {
          chisel3.assert(false.B)
        }
      }
      it("should enable all layers by default") {
        intercept[svsim.Simulation.UnexpectedEndOfMessages.type] {
          simulate(new Foo) { dut =>
            dut.clock.step()
          }
        }
      }
      it("should enable all layers when provied with EnableAll") {
        intercept[svsim.Simulation.UnexpectedEndOfMessages.type] {
          simulate(new Foo, layerControl = LayerControl.EnableAll) { dut =>
            dut.clock.step()
          }
        }
      }
      it("should disable all layers when provided with DisableAll") {
        simulate(new Foo, layerControl = LayerControl.DisableAll) { dut =>
          dut.clock.step()
        }
      }
      it("should enable specific layers with Enable") {
        intercept[svsim.Simulation.UnexpectedEndOfMessages.type] {
          simulate(new Foo, layerControl = LayerControl.Enable(A)) { dut =>
            dut.clock.step()
          }
        }
      }
    }
  }
}
