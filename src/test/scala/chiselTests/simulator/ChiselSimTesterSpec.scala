package chiselTests.simulator

import chisel3._
import chisel3.util._
import chisel3.simulator._

import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.must.Matchers

class ChiselSimTesterSpec extends AnyFunSpec with ChiselSimTester with Matchers {
  describe("ChiselSimTester") {
    it("runs GCD correctly") {
      test(new GCD()) { gcd =>
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

    it("runs GCD and generates VCD") {
      val traceFileName = "trace.vcd"

      val workdir = test(
        new GCD(),
        ChiselSimSettings.verilatorBackend(TraceStyle.Vcd(traceFileName), resetWorkspace = true)
      ) { gcd =>
        gcd.io.a.poke(24.U)
        gcd.io.b.poke(36.U)
        gcd.io.loadValues.poke(1.B)
        gcd.clock.step()
        gcd.io.loadValues.poke(0.B)
        gcd.clock.stepUntil(sentinelPort = gcd.io.resultIsValid, sentinelValue = 1, maxCycles = 10)
        gcd.io.resultIsValid.expect(true.B)
        gcd.io.result.expect(12)
      }

      val generatedVcd = os.FilePath(workdir).resolveFrom(os.pwd) / traceFileName
      println(s"generatedVcd: $generatedVcd")
      assert(os.exists(generatedVcd), s"File not found: $generatedVcd")
    }

    it("tests a FIFO") {
      val traceFileName = "trace.vcd"
      val settings = ChiselSimSettings.verilatorBackend(TraceStyle.Vcd(traceFileName), resetWorkspace = true)
      val rand = scala.util.Random

      test(new Queue(UInt(32.W), 64), settings) { dut =>
        for (numEnqs <- Seq(0, 1, dut.entries - 1, dut.entries) ++ Seq.fill(50)(rand.between(1, dut.entries))) {
          val inputs = Seq.fill(numEnqs)(BigInt(dut.gen.getWidth, rand).U)
          dut.io.deq.valid.expect(0.B)
          dut.io.enq.enqueueSeq(inputs)
          dut.io.count.expect(inputs.length)
          dut.clock.step(1)
          // dut.io.deq.valid.expect(1.B)
          dut.io.deq.expectDequeueSeq(inputs)
          dut.io.count.expect(0)
          dut.io.deq.valid.expect(0.B)
        }
      }

      val generatedVcd = guessWorkDir(settings) / traceFileName

      assert(os.exists(generatedVcd), s"File not found: $generatedVcd")
    }

    it("should handle failure") {
      val traceFileName = "trace.vcd"
      val settings = ChiselSimSettings.verilatorBackend(traceStyle = TraceStyle.Vcd(traceFileName))
      val rand = scala.util.Random

      val thrown = the[PeekPokeAPI.FailedExpectationException[_]] thrownBy {
        test(new Queue(UInt(32.W), 64), settings) { dut =>
          val inputs = Seq.fill(10)(BigInt(dut.gen.getWidth, rand).U)
          dut.io.enq.enqueueSeq(inputs)

          dut.io.deq.expectDequeueSeq(inputs.reverse)
        }
      }

      thrown.getMessage must include("Failed Expectation")
      (thrown.getMessage must include).regex(
        """ @\[src/test/scala/chiselTests/simulator/ChiselSimTesterSpec\.scala \d+:\d+\]"""
      )
      thrown.getMessage must include("dut.io.deq.expectDequeueSeq(inputs.reverse)")

      val generatedVcd = guessWorkDir(settings) / traceFileName

      assert(os.exists(generatedVcd), s"File not found: $generatedVcd")
    }

    it("produce legal trace file even with failures") {
      val traceFileName = "trace.fst"
      val settings =
        ChiselSimSettings.verilatorBackend(traceStyle = TraceStyle.Fst(traceFileName), resetWorkspace = true)
      val rand = scala.util.Random

      val thrown = the[PeekPokeAPI.FailedExpectationException[_]] thrownBy {
        test(new Queue(UInt(32.W), 64), settings) { dut =>
          val inputs = Seq.fill(15)(BigInt(dut.gen.getWidth, rand))
          dut.io.enq.enqueueSeq(inputs.map(_.U))

          dut.io.deq.expectDequeueSeq(inputs.map(i => (i + 1).U))
        }
      }

      thrown.getMessage must include("Failed Expectation")
      (thrown.getMessage must include).regex(
        """ @\[src/test/scala/chiselTests/simulator/ChiselSimTesterSpec\.scala \d+:\d+\]"""
      )
      thrown.getMessage must include("dut.io.deq.expectDequeueSeq(")

      val generatedTrace = guessWorkDir(settings) / traceFileName

      println(s"generatedTrace: $generatedTrace")

      assert(os.exists(generatedTrace), s"File not found: $generatedTrace")

      // FIXME: generated FST is malformed!
      // // We probably need external dependencies to check the generated trace, e.g. fst2vcd from GtkWave
      // os.proc("fst2vcd", generatedTrace).call(check=true)
    }
  }
}
