package chiselTests.simulator

import chisel3._
import chisel3.simulator._
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.must.Matchers
import org.scalatest.matchers.should.Matchers.convertToAnyShouldWrapper
import svsim._
import scala.io.Source

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

    it("reports failed expects correctly") {
      val simulator = new VerilatorSimulator("test_run_dir/simulator/GCDSimulator")
      val thrown = the[PeekPokeAPI.FailedExpectationException[_]] thrownBy {
        simulator
          .simulate(new GCD()) { module =>
            import PeekPokeAPI._
            val gcd = module.wrapped
            gcd.io.a.poke(24.U)
            gcd.io.b.poke(36.U)
            gcd.io.loadValues.poke(1.B)
            gcd.clock.step()
            gcd.io.loadValues.poke(0.B)
            gcd.clock.step(10)
            gcd.io.result.expect(5)
          }
          .result
      }
      thrown.getMessage must include("Observed value '12' != 5.")
      (thrown.getMessage must include).regex(
        """ @\[src/test/scala/chiselTests/simulator/SimulatorSpec\.scala \d+:\d+\]"""
      )
      thrown.getMessage must include("gcd.io.result.expect(5)")
      thrown.getMessage must include("                    ^")
    }

    it("simulate a circuit with zero-width ports") {
      val width = 0
      // Run a simulation with zero width foo
      new VerilatorSimulator("test_run_dir/simulator/foo_zero_width")
        .simulate(new OptionalIOModule(n = width)) { module =>
          import PeekPokeAPI._
          val dut = module.wrapped
          dut.clock.step(2)
          dut.clock.step(10)
        }
        .result

      // Check the testbench sv does not contain dut.foo and dut.out
      val tbSource = Source.fromFile("test_run_dir/simulator/foo_zero_width/generated-sources/testbench.sv")
      val tbSV = tbSource.mkString
      tbSource.close()
      // Check IO ports
      (tbSV should not).include("dut.foo")
      (tbSV should not).include("dut.bar")
      (tbSV should not).include("dut.out")
      (tbSV should not).include("dut.emptyBundle")
      (tbSV should not).include("dut.bundle_x")

      val source = Source.fromFile("test_run_dir/simulator/foo_zero_width/primary-sources/OptionalIOModule.sv")
      val actualSV = source.mkString
      source.close()
      (actualSV should not).include("foo")
      (actualSV should not).include("bar")
      (actualSV should not).include("myReg")
      (actualSV should not).include("output [7:0] out")
      (actualSV should not).include("emptyBundle")
      (actualSV should not).include("bundle_x")
    }

    it("simulate a circuit with non zero-width ports") {
      val width = 8
      // Run a simulation with zero width foo
      new VerilatorSimulator("test_run_dir/simulator/foo_non_zero_width")
        .simulate(new OptionalIOModule(n = width)) { module =>
          import PeekPokeAPI._
          val dut = module.wrapped
          dut.clock.step(2)
          dut.clock.step(10)
        }
        .result

      // Check the testbench sv does not contain dut.foo and dut.out
      val tbSource = Source.fromFile("test_run_dir/simulator/foo_non_zero_width/generated-sources/testbench.sv")
      val tbSV = tbSource.mkString
      tbSource.close()
      // Check IO ports
      tbSV should include("[$bits(dut.foo)-1:0] foo")
      tbSV should include("[$bits(dut.bar)-1:0] bar")
      tbSV should include("[$bits(dut.out)-1:0] out")
      (tbSV should not).include("emptyBundle")
      tbSV should include("[$bits(dut.bundle_x)-1:0] bundle_x")

      val source = Source.fromFile("test_run_dir/simulator/foo_non_zero_width/primary-sources/OptionalIOModule.sv")
      val actualSV = source.mkString
      source.close()
      actualSV should include("foo")
      actualSV should include("bar")
      actualSV should include("myReg")
      actualSV should include("output [7:0] out")
      (actualSV should not).include("emptyBundle")
      actualSV should include("bundle_x")
    }

    it("support peeking and poking FlatIO ports and other views of ports") {
      import chisel3.experimental.dataview._
      class SimpleModule extends Module {
        val io = FlatIO(new Bundle {
          val in = Input(UInt(8.W))
          val out = Output(UInt(8.W))
        })
        val viewOfClock = clock.viewAs[Clock]
        val delay = RegNext(io.in)
        io.out := delay
      }
      new VerilatorSimulator("test_run_dir/simulator/flat_io_ports")
        .simulate(new SimpleModule) { module =>
          import PeekPokeAPI._
          val dut = module.wrapped
          dut.io.in.poke(12.U)
          dut.viewOfClock.step(1)
          dut.io.out.peek()
          dut.io.out.expect(12)
        }
        .result
    }
  }
}
