// SPDX-License-Identifier: Apache-2.0

package chiselTests.simulator.scalatest

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.{RunUntilFinished, RunUntilSuccess}
import chisel3.simulator.{FailedExpectationException, HasSimulator, MacroText, Randomization, Settings}
import chisel3.testing.HasTestingDirectory
import chisel3.testing.scalatest.{FileCheck, TestingDirectory}
import chisel3.util.Counter
import chisel3.util.circt.{PlusArgsTest, PlusArgsValue}
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers

import java.nio.file.FileSystems
import scala.reflect.io.Directory

class ChiselSimSpec extends AnyFunSpec with Matchers with ChiselSim with FileCheck {

  describe("scalatest.ChiselSim") {

    it("should work correctly for poke and expect") {
      class Foo extends RawModule {
        val a = IO(Input(Bool()))
        val b = IO(Output(Bool()))

        b :<= !a
      }

      info("poke and expect work")
      simulateRaw(new Foo) { foo =>
        foo.a.poke(true.B)
        foo.b.expect(false.B)

        foo.a.poke(false.B)
        foo.b.expect(true.B)
      }

      info("an expect throws an exception")
      intercept[FailedExpectationException[_]] {
        simulateRaw(new Foo) { foo =>
          foo.a.poke(false.B)
          foo.b.expect(false.B)
        }
      }
    }

    it("should error if an expect fails") {
      val message = intercept[Exception] {
        simulate {
          new Module {
            val a = IO(Output(Bool()))
            a :<= false.B
          }
        } { _.a.expect(true.B) }
      }.getMessage
      message.fileCheck() {
        """|CHECK:      Failed Expectation
           |CHECK-NEXT: ---
           |CHECK-NEXT: Observed value: '0'
           |CHECK-NEXT: Expected value: '1'
           |CHECK:      ---
           |""".stripMargin
      }
    }

    it("should error if a chisel3.assert fires during the simulation") {
      class Foo extends Module {
        chisel3.assert(false.B, "foo assertion")
      }

      val message = intercept[Exception] {
        simulate(new Foo) { foo =>
          foo.clock.step(4)
        }
      }.getMessage

      message.fileCheck()(
        """|CHECK:      One or more assertions failed during Chiselsim simulation
           |CHECK-NEXT: ---
           |CHECK-NEXT: The following assertion failures were extracted from the log file:
           |CHECK:      lineNo  line
           |CHECK-NEXT: ---
           |CHECK-NEXT:      0  [{{[0-9]+}}] %Error:
           |CHECK:      For more information, see the complete log file:
           |CHECK:        build/chiselsim/ChiselSimSpec/scalatest.ChiselSim/should-error-if-a-chisel3.assert-fires-during-the-simulation/workdir-verilator/simulation-log.txt
           |CHECK-NEXT: ---
           |""".stripMargin
      )
    }

    it("should error if an ltl.AssertProperty fires during the simulation") {
      class Foo extends Module {
        ltl.AssertProperty(false.B)
      }

      val message = intercept[Exception] {
        simulate(new Foo) { foo =>
          foo.clock.step(4)
        }
      }.getMessage

      message.fileCheck()(
        """|CHECK:      One or more assertions failed during Chiselsim simulation
           |CHECK-NEXT: ---
           |CHECK-NEXT: The following assertion failures were extracted from the log file:
           |CHECK:      lineNo  line
           |CHECK-NEXT: ---
           |CHECK-NEXT:      0  [{{[0-9]+}}] %Error:
           |CHECK:      For more information, see the complete log file:
           |CHECK:        build/chiselsim/ChiselSimSpec/scalatest.ChiselSim/should-error-if-an-ltl.AssertProperty-fires-during-the-simulation/workdir-verilator/simulation-log.txt
           |CHECK-NEXT: ---
           |""".stripMargin
      )
    }

    it("should allow for customization of macros during simulation") {
      class Foo extends RawModule {
        val a, b, c = IO(Input(Bool()))
      }

      val settings = Settings
        .defaultRaw[Foo]
        .copy(
          assertVerboseCond = Some(MacroText.Signal(_.a)),
          printfCond = Some(MacroText.Signal(_.b)),
          stopCond = Some(MacroText.NotSignal(_.c))
        )

      simulateRaw(new Foo, settings = settings) { _ => }

      io.Source
        .fromFile(
          FileSystems
            .getDefault()
            .getPath(implicitly[HasTestingDirectory].getDirectory.toString, "workdir-verilator", "Makefile")
            .toFile
        )
        .mkString
        .fileCheck()(
          """|CHECK:      '+define+ASSERT_VERBOSE_COND=svsimTestbench.a'
             |CHECK-NEXT: '+define+PRINTF_COND=svsimTestbench.b'
             |CHECK-NEXT: '+define+STOP_COND=!svsimTestbench.c'
             |""".stripMargin
        )
    }

    it("should allow for a user to customize the build directory") {
      class Foo extends Module {
        stop()
      }

      /** An implementation that always writes to the subdirectory "test_run_dir/<class-name>/foo/" */
      implicit val fooDirectory = new HasTestingDirectory {
        override def getDirectory =
          FileSystems.getDefault().getPath("test_run_dir", "foo")
      }

      val directory = Directory(FileSystems.getDefault().getPath("test_run_dir", "foo").toFile())
      directory.deleteRecursively()

      simulate(new Foo()) { _ => }(
        hasSimulator = implicitly[HasSimulator],
        testingDirectory = fooDirectory,
        implicitly,
        implicitly,
        implicitly,
        implicitly
      )

      info(s"found expected directory: '$directory'")
      assert(directory.exists)
      assert(directory.isDirectory)

      val allFiles = directory.deepFiles.toSeq.map(_.toString).toSet
      for (
        file <- Seq(
          "test_run_dir/foo/workdir-verilator/Makefile",
          "test_run_dir/foo/primary-sources/Foo.sv"
        )
      ) {
        info(s"found expected file: '$file'")
        allFiles should contain(file)
      }
    }

    it("should allow the user to change the subdirectory on SimulatorAPI methods") {
      class Foo extends Module {
        stop()
      }

      var file = FileSystems
        .getDefault()
        .getPath(implicitly[HasTestingDirectory].getDirectory.toString, "foo", "workdir-verilator", "Makefile")
        .toFile
      file.delete()
      simulate(new Foo, subdirectory = Some("foo")) { _ => }
      info(s"$file exists")
      file should (exist)

      file = FileSystems
        .getDefault()
        .getPath(implicitly[HasTestingDirectory].getDirectory.toString, "bar", "workdir-verilator", "Makefile")
        .toFile
      file.delete()
      simulateRaw(new Foo, subdirectory = Some("bar")) { _ => }
      info(s"$file exists")
      file should (exist)
    }

    // Return a Verilator `HasSimulator` that will dump waves to `trace.vcd`.
    def verilatorWithVcd = HasSimulator.simulators
      .verilator(verilatorSettings =
        svsim.verilator.Backend.CompilationSettings.default.withTraceStyle(
          Some(
            svsim.verilator.Backend.CompilationSettings
              .TraceStyle(
                svsim.verilator.Backend.CompilationSettings.TraceKind.Vcd,
                traceUnderscore = true,
                maxArraySize = Some(1024),
                maxWidth = Some(1024),
                traceDepth = Some(1024)
              )
          )
        )
      )

    // Return a Verilator `HasSimulator` that will dump waves to `trace.fst`.
    def verilatorWithFst = HasSimulator.simulators
      .verilator(verilatorSettings =
        svsim.verilator.Backend.CompilationSettings.default.withTraceStyle(
          Some(
            svsim.verilator.Backend.CompilationSettings
              .TraceStyle(
                svsim.verilator.Backend.CompilationSettings.TraceKind.Fst(Some(2)),
                traceUnderscore = true,
                maxArraySize = Some(1024),
                maxWidth = Some(1024),
                traceDepth = Some(1024)
              )
          )
        )
      )

    it("should dump a VCD waveform when traceStyle is Vcd and enableWaves is used") {

      implicit val verilator = verilatorWithVcd

      class Foo extends Module {
        stop()
      }

      val waveFile = FileSystems
        .getDefault()
        .getPath(implicitly[HasTestingDirectory].getDirectory.toString, "workdir-verilator", "trace.vcd")
        .toFile

      waveFile.delete

      simulateRaw(new Foo) { _ =>
        enableWaves()
      }

      info(s"$waveFile exists")
      waveFile should (exist)
      waveFile.length() should be > 1L
    }

    it("should dump a FST waveform when traceStyle is Fst and enableWaves is used") {

      implicit val vaerilator = verilatorWithFst

      class Foo extends Module {
        stop()
      }

      val waveFile = FileSystems
        .getDefault()
        .getPath(implicitly[HasTestingDirectory].getDirectory.toString, "workdir-verilator", "trace.fst")
        .toFile

      waveFile.delete

      simulateRaw(new Foo) { _ =>
        enableWaves()
      }

      info(s"$waveFile exists")
      waveFile should (exist)
      waveFile.length() should be > 1L
    }

    it("should dump a waveform using ChiselSim settings") {

      implicit val vaerilator = verilatorWithVcd

      class Foo extends Module {
        stop()
      }

      val vcdFile = FileSystems
        .getDefault()
        .getPath(implicitly[HasTestingDirectory].getDirectory.toString, "workdir-verilator", "trace.vcd")
        .toFile

      vcdFile.delete

      simulate(new Foo, settings = Settings.default.copy(enableWavesAtTimeZero = true)) { _ => }

      info(s"$vcdFile exists")
      vcdFile should (exist)
    }

    it("should support passing '$value$plusargs' and '$test$plusargs'") {

      class Foo extends Module {
        val value = IO(Output(Bool()))
        val test = IO(Output(Bool()))

        value :<= PlusArgsValue(chiselTypeOf(value), "value=%d", false.B)
        test :<= PlusArgsTest("test")
      }

      simulateRaw(
        new Foo,
        settings = Settings.default.copy(
          plusArgs = Seq(
            new svsim.PlusArg("value", Some("1")),
            new svsim.PlusArg("test", None)
          )
        )
      ) { dut =>
        dut.value.expect(true.B)
        dut.test.expect(true.B)
      }

    }

    class RandomizationTest extends Module {
      val in = IO(Input(UInt(32.W)))
      val addr = IO(Input(UInt(1.W)))
      val we = IO(Input(Bool()))

      val r = IO(Output(UInt(32.W)))
      val cm = IO(Output(UInt(32.W)))
      val sm = IO(Output(UInt(32.W)))

      private val reg = Reg(UInt(32.W))
      private val cmem = Mem(2, UInt(32.W))
      private val smem = SyncReadMem(2, UInt(32.W))

      when(we) {
        reg :<= in
        cmem.write(addr, in)
        smem.write(addr, in)
      }

      r :<= reg
      cm :<= cmem.read(addr)
      sm :<= smem.read(addr)
    }

    it("should have randomization on by default") {

      simulate(new RandomizationTest) { dut =>
        dut.clock.step(2)

        val regValue = dut.r.peekValue().asBigInt
        info(s"register is not zero, has value $regValue")
        regValue should not be (0)

        val cmemValue = dut.cm.peekValue().asBigInt
        info(s"combinational read memory index zero is not zero, has value $cmemValue")
        cmemValue should not be (0)

        val smemValue = dut.sm.peekValue().asBigInt
        info(s"sequential read memory index zero is not zero, has value $smemValue")
        cmemValue should not be (0)
      }

    }

    it("uninitialized randomization should result in zeros (for Verilator)") {

      simulate(new RandomizationTest, settings = Settings.default.copy(randomization = Randomization.uninitialized)) {
        dut =>
          dut.clock.step(2)

          val regValue = dut.r.peekValue().asBigInt
          info(s"register is zero")
          regValue should be(0)

          val cmemValue = dut.cm.peekValue().asBigInt
          info(s"combinational read memory index zero is zero")
          cmemValue should be(0)

          val smemValue = dut.sm.peekValue().asBigInt
          info(s"sequential read memory index zero is zero")
          cmemValue should be(0)
      }

    }

    it("the randomization value should be user-overridable") {

      simulate(
        new RandomizationTest,
        settings = Settings.default.copy(randomization = Randomization.random.copy(randomValue = Some("{32'd42}")))
      ) { dut =>
        dut.clock.step(2)

        val regValue = dut.r.peekValue().asBigInt
        info(s"register is 42")
        regValue should be(42)

        val cmemValue = dut.cm.peekValue().asBigInt
        info(s"combinational read memory index zero is 42")
        cmemValue should be(42)

        val smemValue = dut.sm.peekValue().asBigInt
        info(s"sequential read memory index zero is 42")
        cmemValue should be(42)
      }

    }

    it("should allow setting the frequency to 1GHz") {

      import svsim.{CommonCompilationSettings, CommonSettingsModifications}

      class Foo extends Module {

        when(Counter(true.B, 8)._2) {
          stop()
        }

      }

      simulate(new Foo)(RunUntilFinished(10, period = 10))

    }

    // This is checking that `.fir` files in the output aren't included in the compile.
    it("not fail to compile with the --dump-fir option") {

      class Foo extends Module {
        stop()
      }

      simulate(new Foo, chiselOpts = Array("--dump-fir"))(RunUntilFinished(2))

    }

    it("should handle non-zero delays in external modules with SystemVerilog sources elegantly") {
      import chisel3.util.HasBlackBoxInline

      trait DelayedIO {
        val in = IO(Input(UInt(1.W)))
        val delayedIn = IO(Output(UInt(1.W)))
        val delayedInitial = IO(Output(UInt(1.W)))
      }

      class Delayed extends ExtModule with DelayedIO {

        setInline(
          "Delayed.sv",
          """
            |`timescale 1 ps / 1 ps
            |
            |module Delayed(
            |  input in,
            |  output reg delayedIn,
            |  output reg delayedInitial
            |);
            |
            |  always @(in) begin
            |    delayedIn <= #1500 in;
            |  end
            |
            |  initial begin
            |    delayedInitial = 1'b0;
            |    #800
            |    delayedInitial = 1'b1;
            |    #800
            |    delayedInitial = 1'b0;
            |  end
            |
            |endmodule
            |""".stripMargin
        )
      }

      class Foo extends Module with DelayedIO {
        val delayed = Module(new Delayed)
        delayed.in :<= in
        delayedIn :<= delayed.delayedIn
        delayedInitial :<= delayed.delayedInitial

        // Some simple logic using the clock
        val counter = RegInit(0.U(8.W))
        counter :<= counter + 1.U
      }

      // Temporarily modify backend settings for Verilator to enable `--timing`
      implicit def backendSettingsModifications: svsim.BackendSettingsModifications = {
        case conf: svsim.verilator.Backend.CompilationSettings =>
          svsim.verilator.Backend.CompilationSettings.default
            .withDisableFatalExitOnWarnings(true)
            .withTiming(Some(svsim.verilator.Backend.CompilationSettings.Timing.TimingEnabled))
        case conf => conf
      }

      simulate(new Foo)({ dut =>
        dut.in.poke(1.U)
        dut.delayedIn.expect(0.U)
        dut.delayedInitial.expect(0.U)
        dut.clock.step(1, 1000)

        dut.delayedIn.expect(0.U)
        dut.delayedInitial.expect(1.U)
        dut.clock.step(1, 1000)

        dut.in.poke(0.U)
        dut.delayedIn.expect(1.U)
        dut.delayedInitial.expect(0.U)
        dut.clock.step(2, 1000)

        dut.delayedIn.expect(0.U)
      })
    }

  }

  describe("ChiselSim RunUntilSuccess stimulus") {

    class SuccessAfterFourCycles extends Module {
      val success = IO(Output(Bool()))
      success :<= Counter(true.B, 4)._2
    }

    it("should report success for a passing Module") {
      simulate(new SuccessAfterFourCycles)(RunUntilSuccess(maxCycles = 8, getSuccess = _.success))
    }

    it("should throw an exception for a failing Module") {
      intercept[chisel3.simulator.Exceptions.Timeout] {
        simulate(new SuccessAfterFourCycles)(RunUntilSuccess(maxCycles = 2, getSuccess = _.success))
      }
    }

  }

  describe("ChiselSim user errors") {

    it("should provide a sane error message if a user pokes an output port") {
      class Foo extends RawModule {
        val a = IO(Output(Bool()))
        a :<= DontCare
      }
      intercept[java.lang.IllegalArgumentException] {
        simulateRaw(new Foo) { dut =>
          dut.a.poke(false.B)
        }
      }.getMessage.fileCheck() { "CHECK: cannot set port 'a' (id: '0') because it is not settable" }
    }

  }

  describe("Specific ChiselSim issues") {

    it("should not hang like in #5128") {
      import scala.concurrent.{Await, Future}
      import scala.concurrent.duration.DurationInt

      class Incrementer extends Module {
        val in = IO(Input(UInt(8.W)))
        val sel = IO(Input(Bool()))
        val out = IO(Output(UInt(9.W)))
        out := Mux(sel, in +& 1.U, in)
      }

      def randomIncrementerTest(
        numberOfTests: Int
      ): Unit = {
        simulate(new Incrementer) { dut =>
          for (i <- 1 to numberOfTests) {
            dut.in.poke(1)
            dut.sel.poke(1)
            dut.out.expect(2)
            dut.sel.poke(0)
            dut.out.expect(1)
          }
        }
      }

      implicit val ec: scala.concurrent.ExecutionContext = scala.concurrent.ExecutionContext.global
      Await.result(Future {
        randomIncrementerTest(3000) // more than 2802 to trigger the issue
      }, 10.seconds)
    }

  }

}
