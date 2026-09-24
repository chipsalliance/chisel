package chiselTests.simulator

import chisel3._

// Import the simulator and the settings
import chisel3.simulator.parametric.ParametricSimulator._
import chisel3.simulator.parametric.simulatorSettings._

import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.must.Matchers

import java.nio.file.{Files, Paths}

class ParametricSimulatorSpec extends AnyFunSpec with Matchers {

  // The testbench function used in this test
  private def gcdTb(gcd: => GCD): Unit = {
    gcd.io.a.poke(24.U)
    gcd.io.b.poke(36.U)
    gcd.io.loadValues.poke(1.B)
    gcd.clock.step()
    gcd.io.loadValues.poke(0.B)
    gcd.clock.stepUntil(sentinelPort = gcd.io.resultIsValid, sentinelValue = 1, maxCycles = 10)
    gcd.io.resultIsValid.expect(true.B)
    gcd.io.result.expect(12)
  }

  describe("ParametricSimulator runs") {

    it("runs GCD correctly without settings") {
      simulate(new GCD())(gcd => gcdTb(gcd))
    }
  }

  describe("ParametricSimulator: VCD Trace") {
    it("runs GCD with VCD trace file") {
      simulate(new GCD(), Seq(VcdTrace))(gcd => gcdTb(gcd))
      assert(Files.exists(Paths.get("test_run_dir/GCD/ParametricSimulator/defaultSimulation/trace.vcd")))
    }

    it("runs GCD with VCD underscore trace") {
      simulate(new GCD(), Seq(VcdTraceWithUnderscore))(gcd => gcdTb(gcd))
      assert(Files.exists(Paths.get("test_run_dir/GCD/ParametricSimulator/defaultSimulation/trace_underscore.vcd")))
    }

    it("runs GCD with VCD trace file with name") {
      simulate(new GCD(), Seq(VcdTrace, NameTrace("gcdTbTraceName")))(gcd => gcdTb(gcd))
      assert(Files.exists(Paths.get("test_run_dir/GCD/ParametricSimulator/defaultSimulation/gcdTbTraceName.vcd")))
    }

    it("runs underscore when VcdTrace and VcdTraceWithUnderscore are used") {
      simulate(new GCD(), Seq(VcdTrace, VcdTraceWithUnderscore))(gcd => gcdTb(gcd))
      assert(Files.exists(Paths.get("test_run_dir/GCD/ParametricSimulator/defaultSimulation/trace_underscore.vcd")))

      simulate(new GCD(), Seq(VcdTraceWithUnderscore, VcdTrace))(gcd => gcdTb(gcd))
      assert(Files.exists(Paths.get("test_run_dir/GCD/ParametricSimulator/defaultSimulation/trace_underscore.vcd")))
    }

    it("runs GCD with VCD trace file with name and VCD underscore trace") {
      simulate(new GCD(), Seq(VcdTrace, VcdTraceWithUnderscore, NameTrace("gcdTb1")))(gcd => gcdTb(gcd))
      assert(Files.exists(Paths.get("test_run_dir/GCD/ParametricSimulator/defaultSimulation/gcdTb1_underscore.vcd")))

      simulate(new GCD(), Seq(VcdTraceWithUnderscore, VcdTrace, NameTrace("gcdTb2")))(gcd => gcdTb(gcd))
      assert(Files.exists(Paths.get("test_run_dir/GCD/ParametricSimulator/defaultSimulation/gcdTb2_underscore.vcd")))

      simulate(new GCD(), Seq(NameTrace("gcdTb3"), VcdTraceWithUnderscore, VcdTrace))(gcd => gcdTb(gcd))
      assert(Files.exists(Paths.get("test_run_dir/GCD/ParametricSimulator/defaultSimulation/gcdTb3_underscore.vcd")))
    }
  }

  describe("ParametricSimulator: SaveWorkdir") {

    it("uses a name for the simulation") {
      simulate(new GCD(), Seq(VcdTrace, NameTrace("gcdTb1")), simName = "use_a_name_for_the_simulation")(gcd =>
        gcdTb(gcd)
      )
      assert(Files.exists(Paths.get("test_run_dir/GCD/ParametricSimulator/use_a_name_for_the_simulation/gcdTb1.vcd")))
    }

    it("save the workdir with a name") {
      simulate(new GCD(), Seq(VcdTrace, SaveWorkdirFile("myWorkdir")))(gcd => gcdTb(gcd))
      assert(Files.exists(Paths.get("test_run_dir/GCD/ParametricSimulator/defaultSimulation/myWorkdir")))
    }
  }

  describe("ParametricSimulator: Custom Firtool compilation") {
    it("uses firtool args") {
      simulate(new GCD(), Seq(WithFirtoolArgs(Seq("-g", "--emit-hgldd")), SaveWorkdirFile("myWorkdir2")))(gcd =>
        gcdTb(gcd)
      )
      assert(Files.exists(Paths.get("test_run_dir/GCD/ParametricSimulator/defaultSimulation/myWorkdir2")))
      assert(
        Files.exists(
          Paths.get("test_run_dir/GCD/ParametricSimulator/defaultSimulation/myWorkdir2/support-artifacts/GCD.dd")
        )
      )
    }

  }

  describe("ParametricSimulator Exceptions") {

    it("throws an exception when NameTrace is used without VcdTrace or VcdTraceWithUnderscore") {
      intercept[Exception] {
        simulate(new GCD(), Seq(NameTrace("")))(gcd => gcdTb(gcd))
      }
    }

    it("throws an exception with two or more NameTrace") {
      intercept[Exception] {
        simulate(new GCD(), Seq(VcdTrace, NameTrace("a"), NameTrace("b")))(gcd => gcdTb(gcd))
      }
    }

    it("throws an exception with two or more SaveWorkdir") {
      intercept[Exception] {
        simulate(new GCD(), Seq(SaveWorkdir, SaveWorkdir))(gcd => gcdTb(gcd))
      }
    }

    it("throws an exception with two or more SaveWorkdirFile") {
      intercept[Exception] {
        simulate(new GCD(), Seq(SaveWorkdirFile("a"), SaveWorkdirFile("b")))(gcd => gcdTb(gcd))
      }
    }

  }
}
