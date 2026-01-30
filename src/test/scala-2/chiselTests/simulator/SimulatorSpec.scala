package chiselTests.simulator

import chisel3._
import chisel3.layer.{block, Convention, Layer, LayerConfig}
import chisel3.simulator._
import chisel3.util.circt.dpi._
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.must.Matchers
import org.scalatest.matchers.should.Matchers.convertToAnyShouldWrapper
import svsim._

class VerilatorSimulator(val workspacePath: String) extends Simulator[verilator.Backend] {
  val backend = verilator.Backend.initializeFromProcessEnvironment()
  val tag = "verilator"
  val commonCompilationSettings = CommonCompilationSettings()
  val backendSpecificCompilationSettings = verilator.Backend.CompilationSettings.default
}

object SimulatorSpec {
  class LinkLibraryTest extends RawModule {
    val a = IO(Output(UInt(32.W)))
    val b = IO(Output(UInt(32.W)))
    val c = IO(Output(UInt(32.W)))
    // The following DPI functions are implemented in
    // `src/test/resources/chisel3/simulator/linkLib*.c`. The build system
    // compiles these into shared libraries and makes them available to be linked
    // into the simulator above.
    a := RawUnclockedNonVoidFunctionCall("magicFuncA", UInt(32.W))(true.B)
    b := RawUnclockedNonVoidFunctionCall("magicFuncB", UInt(32.W))(true.B)
    c := RawUnclockedNonVoidFunctionCall("magicFuncC", UInt(32.W))(true.B)
  }
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
          gcd.clock.step(2)
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
      val thrown = the[FailedExpectationException[_]] thrownBy {
        simulator
          .simulate(new GCD()) { module =>
            import PeekPokeAPI._
            val gcd = module.wrapped
            gcd.clock.step(2)
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
      thrown.getMessage must include("observed value 12 != 5")
      (thrown.getMessage must include).regex(
        """ @\[src/test/scala-2/chiselTests/simulator/SimulatorSpec\.scala:\d+:\d+\]"""
      )
      thrown.getMessage must include("gcd.io.result.expect(5)")
      thrown.getMessage must include("                    ^")
    }

    it("runs a design that includes an external module") {
      class Bar extends ExtModule {
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

      class Baz extends ExtModule {
        val a = IO(Output(Bool()))
        addResource("/chisel3/simulator/Baz.sv")
      }

      class Qux extends ExtModule {
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

    it("runs a design with debug mode (-g) and --strip-debug-info") {
      import circt.stage.ChiselStage

      class Bar extends Module {
        val a = IO(Input(Bool()))
        val b = IO(Input(Bool()))
        val out = IO(Output(Bool()))

        out := a & b
      }

      // Check now the debug info is stripped
      val expectedSV = ChiselStage.emitSystemVerilog(new Bar, firtoolOpts = Array("--strip-debug-info", "-g"))

      new VerilatorSimulator("test_run_dir/simulator/bar_debug_mode")
        .simulate(new Bar, firtoolOpts = Array("-strip-debug-info", "-g")) { module =>
          import PeekPokeAPI._
          val bar = module.wrapped

          bar.a.poke(false.B)
          bar.b.poke(false.B)
          bar.out.expect(false.B)
          bar.clock.step()

          bar.a.poke(true.B)
          bar.b.poke(false.B)
          bar.out.expect(false.B)
          bar.clock.step()

          bar.a.poke(true.B)
          bar.b.poke(true.B)
          bar.out.expect(true.B)
          bar.clock.step()
        }
        .result

      // Check the expected SV and the generated SV are the same.
      val source = io.Source.fromFile("test_run_dir/simulator/bar_debug_mode/primary-sources/Bar.sv")
      val actualSV = source.mkString
      expectedSV should include(actualSV)
      source.close()

    }

    it("simulate a circuit with zero-width ports") {
      val width = 0
      // Run a simulation with zero width foo
      new VerilatorSimulator("test_run_dir/simulator/foo_zero_width")
        .simulate(new OptionalIOModule(n = width), firtoolOpts = Array("--strip-debug-info", "-g")) { module =>
          import PeekPokeAPI._
          val dut = module.wrapped
          dut.clock.step(2)
          dut.clock.step(10)
        }
        .result

      // Check the testbench sv does not contain dut.foo and dut.out
      val tbSource = io.Source.fromFile("test_run_dir/simulator/foo_zero_width/generated-sources/testbench.sv")
      val tbSV = tbSource.mkString
      tbSource.close()
      // Check IO ports
      (tbSV should not).include("dut.foo")
      (tbSV should not).include("dut.bar")
      (tbSV should not).include("dut.out")
      (tbSV should not).include("dut.emptyBundle")
      (tbSV should not).include("dut.bundle_x")

      val source = io.Source.fromFile("test_run_dir/simulator/foo_zero_width/primary-sources/OptionalIOModule.sv")
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
        .simulate(new OptionalIOModule(n = width), firtoolOpts = Array("--strip-debug-info", "-g")) { module =>
          import PeekPokeAPI._
          val dut = module.wrapped
          dut.clock.step(2)
          dut.clock.step(10)
        }
        .result

      // Check the testbench sv does not contain dut.foo and dut.out
      val tbSource = io.Source.fromFile("test_run_dir/simulator/foo_non_zero_width/generated-sources/testbench.sv")
      val tbSV = tbSource.mkString
      tbSource.close()
      // Check IO ports
      tbSV should include("[$bits(dut.foo)-1:0] foo")
      tbSV should include("[$bits(dut.bar)-1:0] bar")
      tbSV should include("[$bits(dut.out)-1:0] out")
      (tbSV should not).include("emptyBundle")
      tbSV should include("[$bits(dut.bundle_x)-1:0] bundle_x")

      val source = io.Source.fromFile("test_run_dir/simulator/foo_non_zero_width/primary-sources/OptionalIOModule.sv")
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
          dut.viewOfClock.step(2)
          dut.io.in.poke(12.U)
          dut.viewOfClock.step(1)
          dut.io.out.peek()
          dut.io.out.expect(12)
        }
        .result
    }

    it("has layers enabled") {
      object AssertLayer extends Layer(LayerConfig.Extract())
      class Foo extends Module {
        val a = IO(Input(Bool()))
        block(AssertLayer) {
          chisel3.assert(a, "a must be true")
        }
      }
      intercept[Exception] {
        new VerilatorSimulator("test_run_dir/simulator/has_layers_enabled")
          .simulate(new Foo) { module =>
            import PeekPokeAPI._
            val dut = module.wrapped
            dut.a.poke(false.B)
            dut.clock.step(1)
          }
          .result
      }.getMessage should include("Assertion failed")
    }

    it("does not compile disabled layers (#4676)") {
      import ltl.Sequence.BoolSequence

      class Foo extends Module {
        val a = IO(Input(Bool()))
        // Verilator does not support s_eventually, which this lowers to.
        block(layers.Verification.Assert.Temporal) {
          ltl.AssertProperty(ltl.Property.eventually(a))
        }
      }

      info("illegal constructs cause compilation failure")
      intercept[Exception] {
        new VerilatorSimulator("test_run_dir/simulator/does_not_compile_disabled_layers-enabledf")
          .simulate(
            new Foo,
            settings = Settings.default[Foo].copy(verilogLayers = LayerControl.EnableAll)
          ) { _ => }
          .result
      }.getMessage() should include("Unsupported: s_eventually")

      info("disabling unsupported constracts causes compilation to succeed using LayerControl.DisableAll")
      new VerilatorSimulator("test_run_dir/simulator/does_not_compile_disabled_layers-disabled")
        .simulate(new Foo, settings = Settings.default[Foo].copy(verilogLayers = LayerControl.DisableAll)) { _ => }
        .result

      info(
        "disabling unsupported constracts causes compilation to succeed using LayerControl.Disable(layers.Verification.Assert.Temporal)"
      )
      new VerilatorSimulator("test_run_dir/simulator/does_not_compile_disabled_layers-disabled")
        .simulate(
          new Foo,
          settings =
            Settings.default[Foo].copy(verilogLayers = LayerControl.Disable(layers.Verification.Assert.Temporal))
        ) { _ => }
        .result

    }

    it("supports link libraries provided by the build system") {
      val simulator = new VerilatorSimulator("test_run_dir/simulator/LinkLibrary")
      val settings = Settings
        .defaultRaw[SimulatorSpec.LinkLibraryTest]
        .withLibraries(Seq("linkLibA", "linkLibB"))
        .withLibraryPaths(Seq(sys.env("LINKLIBC_FULL_PATH")))
      val result = simulator
        .simulate(new SimulatorSpec.LinkLibraryTest(), settings = settings) { module =>
          import PeekPokeAPI._
          val dut = module.wrapped
          (dut.a.peek().litValue, dut.b.peek().litValue, dut.c.peek().litValue)
        }
        .result
      assert(result === (42, 1337, 9001))
    }
  }
}
