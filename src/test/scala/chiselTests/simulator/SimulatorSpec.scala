package chiselTests.simulator

import chisel3._
import chisel3.experimental.ExtModule
import chisel3.simulator._
import chisel3.util.{HasExtModuleInline, HasExtModulePath, HasExtModuleResource}
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.must.Matchers
import svsim._

class VerilatorSimulator(val workspacePath: String) extends SingleBackendSimulator[verilator.Backend] {
  val backend = verilator.Backend.initializeFromProcessEnvironment()
  val tag = "verilator"
  val commonCompilationSettings = CommonCompilationSettings()
  val backendSpecificCompilationSettings = verilator.Backend.CompilationSettings()
}

class SimulatorSpec extends AnyFunSpec with Matchers {
  describe("Chisel Simulator") {
    it("runs GCD correctly") {
      val simulator = new VerilatorSimulator("test_run_dir/simulator/GCDSimulator")
      val result = simulator
        .simulate(new GCD()) { module =>
          val gcd = module.wrapped
          val a = module.port(gcd.io.a)
          val b = module.port(gcd.io.b)
          val loadValues = module.port(gcd.io.loadValues)
          val result = module.port(gcd.io.result)
          val resultIsValid = module.port(gcd.io.resultIsValid)
          val clock = module.port(gcd.clock)
          a.set(24)
          b.set(36)
          loadValues.set(1)
          clock.tick(
            inPhaseValue = 0,
            outOfPhaseValue = 1,
            timestepsPerPhase = 1,
            cycles = 1
          )
          loadValues.set(0)
          clock.tick(
            inPhaseValue = 0,
            outOfPhaseValue = 1,
            timestepsPerPhase = 1,
            maxCycles = 10,
            sentinel = Some(resultIsValid, 1)
          )
          result.get().asBigInt
        }
        .result
      assert(result === 12)
    }

    it("runs GCD correctly with peek/poke") {
      val simulator = new VerilatorSimulator("test_run_dir/simulator/GCDSimulator")
      val result = simulator
        .simulate(new GCD()) { module =>
          import PeekPokeAPI._
          val gcd = module.wrapped
          gcd.io.a.poke(24.U)
          gcd.io.b.poke(36.U)
          gcd.io.loadValues.poke(1.B)
          gcd.clock.step()
          gcd.io.loadValues.poke(0.B)
          gcd.clock.step(10)
          gcd.io.result.expect(12)
          gcd.io.result.peek().litValue
        }
        .result
      assert(result === 12)
    }

    it("runs a design that includes an external module") {
      class Bar extends ExtModule with HasExtModuleInline {
        val a = IO(Output(Bool()))
        setInline(
          "Bar.sv",
          """|module Bar(
             |  output a
             |);
             | assign a = 1'b1;
             |endmodule
             |""".stripMargin
        )
      }

      class Baz extends ExtModule with HasExtModuleResource {
        val a = IO(Output(Bool()))
        addResource("/chisel3/simulator/Baz.sv")
      }

      class Qux extends ExtModule with HasExtModulePath {
        val a = IO(Output(Bool()))
        addPath("src/test/resources/chisel3/simulator/Qux.sv")
      }

      class Foo extends RawModule {
        val a, b, c = IO(Output(Bool()))
        a :<= Module(new Bar).a
        b :<= Module(new Baz).a
        c :<= Module(new Qux).a
      }

      new VerilatorSimulator("test_run_dir/simulator/extmodule")
        .simulate(new Foo) { module =>
          import PeekPokeAPI._
          val foo = module.wrapped
          foo.a.expect(1)
          foo.b.expect(1)
          foo.c.expect(1)
        }
        .result
    }
  }
}
