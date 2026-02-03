package chiselTests.simulator

import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.must.Matchers

import chisel3._
import chisel3.layer.{block, Layer, LayerConfig}
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
      describe("for extract layers") {
        object A extends Layer(LayerConfig.Extract())
        class Foo extends Module {
          block(A) {
            chisel3.assert(false.B)
          }
        }
        it("should enable all layers by default") {
          intercept[Exception] {
            simulate(new Foo) { dut =>
              dut.clock.step()
            }
          }.getMessage must include("Assertion failed")
        }
        it("should enable all layers when provied with EnableAll") {
          intercept[Exception] {
            simulate(new Foo, layerControl = LayerControl.EnableAll) { dut =>
              dut.clock.step()
            }
          }.getMessage must include("Assertion failed")
        }
        it("should disable all layers when provided with Enable()") {
          simulate(new Foo, layerControl = LayerControl.Enable()) { dut =>
            dut.clock.step()
          }
        }
        it("should enable specific layers with Enable") {
          intercept[Exception] {
            simulate(new Foo, layerControl = LayerControl.Enable(A)) { dut =>
              dut.clock.step()
            }
          }.getMessage must include("Assertion failed")
        }
      }
      describe("for inline layers") {
        object A extends Layer(LayerConfig.Inline)
        object B extends Layer(LayerConfig.Inline)
        class Foo extends Module {
          block(A) {
            chisel3.assert(false.B)
          }
        }
        it("should enable all layers by default") {
          intercept[Exception] {
            simulate(new Foo) { dut =>
              dut.clock.step()
            }
          }.getMessage must include("Assertion failed")
        }
        it("should enable all layers when provied with EnableAll") {
          intercept[Exception] {
            simulate(new Foo, layerControl = LayerControl.EnableAll) { dut =>
              dut.clock.step()
            }
          }.getMessage must include("Assertion failed")
        }
        it("should disable all layers when provided with Enable()") {
          simulate(new Foo, layerControl = LayerControl.Enable()) { dut =>
            dut.clock.step()
          }
        }
        it("should enable specific layers with Enable") {
          intercept[Exception] {
            simulate(new Foo, layerControl = LayerControl.Enable(A)) { dut =>
              dut.clock.step()
            }
          }.getMessage must include("Assertion failed")
        }
        it("should error if an enabled layer does not exist") {
          intercept[IllegalArgumentException] {
            simulate(new Foo, layerControl = LayerControl.Enable(B)) { dut =>
              dut.clock.step()
            }
          }.getMessage() must include("cannot enable layer 'B' as it is not one of the defined layers")
        }
      }
    }
  }
}
