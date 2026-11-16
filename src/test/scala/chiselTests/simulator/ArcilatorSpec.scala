// SPDX-License-Identifier: Apache-2.0

package chiselTests.simulator

import chisel3._
import chisel3.simulator.{ChiselSim, Exceptions, HasSimulator}
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers
import java.nio.file.Files
import chisel3.testing.HasTestingDirectory

class ArcilatorSpec extends AnyFunSpec with Matchers with ChiselSim {
  implicit val hasSimulator: HasSimulator = HasSimulator.simulators.arcilator()

  class Pipeline extends Module {
    val in = IO(Input(UInt(129.W)))
    val out = IO(Output(UInt(129.W)))
    val signedIn = IO(Input(SInt(13.W)))
    val signedOut = IO(Output(SInt(13.W)))
    out := RegNext(in, 0.U)
    signedOut := RegNext(signedIn, 0.S)
  }

  describe("arcilator") {
    it("supports wide and signed ports, reset, and repeated clock stepping") {
      simulate(new Pipeline) { dut =>
        dut.out.expect(0.U)
        val value = (BigInt(1) << 128) + 42
        dut.in.poke(value.U)
        dut.signedIn.poke((-123).S)
        dut.clock.step()
        dut.out.expect(value.U)
        dut.signedOut.expect((-123).S)
        dut.in.poke(17.U)
        dut.clock.step(3)
        dut.out.expect(17.U)
        dut.reset.poke(true.B)
        dut.clock.step()
        dut.out.expect(0.U)
      }
    }

    it("evaluates combinational logic after pokes") {
      class Inverter extends RawModule {
        val in = IO(Input(UInt(7.W)))
        val out = IO(Output(UInt(7.W)))
        out := ~in
      }
      simulateRaw(new Inverter) { dut =>
        dut.in.poke(42.U)
        dut.out.expect(85.U)
        dut.in.poke(0.U)
        dut.out.expect(127.U)
      }
    }

    it("reports hardware assertion failures") {
      class Checked extends Module {
        val fail = IO(Input(Bool()))
        chisel3.assert(!fail, "arcilator assertion")
      }
      intercept[Exceptions.AssertionFailed] {
        simulate(new Checked) { dut =>
          dut.fail.poke(true.B)
          dut.clock.step()
        }
      }
    }

    it("supports memories") {
      class Memory extends Module {
        val address = IO(Input(UInt(2.W)))
        val data = IO(Input(UInt(17.W)))
        val write = IO(Input(Bool()))
        val out = IO(Output(UInt(17.W)))
        val memory = Mem(4, UInt(17.W))
        when(write) { memory.write(address, data) }
        out := memory.read(address)
      }
      simulate(new Memory) { dut =>
        dut.address.poke(2.U)
        dut.data.poke(12345.U)
        dut.write.poke(true.B)
        dut.clock.step()
        dut.write.poke(false.B)
        dut.out.expect(12345.U)
      }
    }

    it("supports successful hardware termination") {
      class Finished extends Module {
        val done = IO(Input(Bool()))
        when(done) { stop() }
      }
      simulate(new Finished) { dut =>
        dut.done.poke(true.B)
        dut.clock.step()
      }
    }

    it("writes VCD waveforms with runtime wave control") {
      val directory = Files.createTempDirectory("arcilator-waves")
      implicit val testingDirectory: HasTestingDirectory = new HasTestingDirectory {
        override def getDirectory = directory
      }
      implicit val backendSettings: svsim.BackendSettingsModifications = {
        case settings: svsim.arcilator.Backend.CompilationSettings => settings.copy(enableVcd = true)
        case settings => settings
      }
      simulate(new Pipeline) { dut =>
        enableWaves()
        dut.in.poke(42.U)
        dut.clock.step(2)
        disableWaves()
      }
      val trace = directory.resolve("workdir-arcilator/trace.vcd")
      Files.isRegularFile(trace) should be(true)
      val contents = new String(Files.readAllBytes(trace), java.nio.charset.StandardCharsets.UTF_8)
      contents should include("$enddefinitions $end")
      contents should include("$var wire 129")
      contents should include("#")
    }
  }
}

class ArcilatorChiseltestSpec extends AnyFunSpec with Matchers with chiseltest.ChiselScalatestTester {
  import chiseltest._

  it("selects arcilator via annotations and writes a VCD") {
    class Counter extends Module {
      val out = IO(Output(UInt(8.W)))
      val count = RegInit(0.U(8.W))
      count := count + 1.U
      out := count
    }
    test(new Counter).withAnnotations(Seq(ArcilatorBackendAnnotation, WriteVcdAnnotation)) { dut =>
      dut.out.expect(0.U)
      dut.clock.step(5)
      dut.out.expect(5.U)
    }
  }
}

class ArcilatorCliSpec
    extends AnyFunSpec
    with chisel3.simulator.scalatest.ChiselSim
    with chisel3.simulator.scalatest.Cli.Simulator {
  override protected def defaultCliSimulator = Some(HasSimulator.simulators.arcilator())

  it("selects arcilator and VCD tracing through CLI options") {
    class Passthrough extends Module {
      val in = IO(Input(UInt(8.W)))
      val out = IO(Output(UInt(8.W)))
      out := RegNext(in, 0.U)
    }
    simulate(new Passthrough) { dut =>
      dut.in.poke(42.U)
      dut.clock.step()
      dut.out.expect(42.U)
    }
  }
}
