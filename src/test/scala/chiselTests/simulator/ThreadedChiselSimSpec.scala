package chiselTests.simulator

import chisel3._
import chisel3.simulator._

import org.scalatest.funspec.AnyFunSpec

class ThreadedChiselSimSpec extends AnyFunSpec with ChiselSimTester {

  val rand = scala.util.Random

  it("should work for a FIFO") {
    val debug = true
    val numTests = if (debug) 1 else 137

    test(
      new chisel3.util.Queue(UInt(32.W), 1, pipe = true, flow = false),
      ChiselSimSettings.verilatorBackend(
        resetWorkspace = !debug,
        traceStyle = if (debug) TraceStyle.Vcd() else TraceStyle.NoTrace,
        verboseRun = debug
      )
    ) { dut =>
      for (testCase <- 1 to numTests) {

        val inputs =
          Seq.tabulate(if (debug) dut.entries + 2 else rand.between(0, 7 * dut.entries))(
            if (debug) _.toLong else _ => rand.nextLong(1L << dut.io.enq.bits.getWidth)
          )
        val expected = inputs

        dut.io.deq.valid.expect(false.B)
        dut.io.enq.ready.expect(true.B)

        fork {
          dut.io.enq.enqueueSeq(inputs.map(_.U))
          // for (_ <- 0 until rand.between(0, 3)) {
          //   dut.clock.step()
          // }
          // for (input <- inputs) {
          //   dut.io.enq.bits.poke(input.U)
          //   dut.io.enq.valid.poke(1)
          //   while (dut.io.enq.ready.peekValue().asBigInt != 1) {
          //     dut.clock.step()
          //   }
          //   dut.clock.step()
          //   dut.io.enq.valid.poke(0)
          // }
        }.fork {
          dut.clock.step(1)
          dut.io.deq.expectDequeueSeq(expected.map(_.U))
          // dut.io.count.expect(0)
          //   for (_ <- 0 until rand.between(0, 3)) {
          //     // dut.io.deq.valid.expect(false.B)
          //     dut.clock.step()
          //   }
        }.joinAndStep()
      }
    }
  }

  it("should work for multiple consumer threads") {
    val numTests = 137

    test(
      new SplitMod(4, 32),
      ChiselSimSettings.verilatorBackend(resetWorkspace = true, traceStyle = TraceStyle.Vcd())
    ) { dut =>
      for (testCase <- 1 to numTests) {
        val inputs = Seq.fill(rand.between(1, 200))(rand.nextLong(1L << dut.io.in.bits.getWidth))
        val expected = inputs.groupBy(_ % dut.n)

        def checkOutput(i: Int) = {
          dut.io.out(i).expectDequeueSeq(expected.getOrElse(i, Seq.empty).map(_.U))
          for (_ <- 0 until rand.between(0, 3)) {
            // dut.io.out(i).valid.expect(false.B)
            dut.clock.step()
          }
        }

        fork {
          dut.io.in.enqueueSeq(inputs.map(_.U))
          for (_ <- 0 until rand.between(0, 3)) {
            dut.clock.step()
          }
        }.fork {
          checkOutput(0)
        }.fork {
          checkOutput(1)
        }.fork {
          checkOutput(2)
        }.fork {
          checkOutput(3)
        }.join()
      }
    }
  }

  it("should work for multiple producer threads") {
    val debug = false
    val numTests = if (debug) 1 else 137
    val w = 32

    test(
      new Adder(5, w),
      ChiselSimSettings.verilatorBackend(resetWorkspace = true, traceStyle = TraceStyle.Vcd(), verboseRun = debug)
    ) { dut =>
      def addMod2ToW(a: Long, b: Long) = (a + b) & ((1L << w) - 1)

      for (testCase <- 1 to numTests) {
        val len = rand.between(0, 50)
        val inputs = Seq.fill(dut.n)(Seq.tabulate(len)(if (debug) _.toLong else _ => rand.between(0L, 1L << w)))
        val expected = inputs.transpose.map(_.reduceOption(addMod2ToW).getOrElse(0L))

        if (debug) {
          println(s"inputs: $inputs")
          println(s"inputGroups:\n${inputs.transpose.map(_.map(_.toHexString).mkString(", ")).mkString("\n")}")
          println(s"expected: $expected")
        }

        fork {
          dut.io.in(0).enqueueSeq(inputs(0).map(_.U))
        }.fork {
          dut.io.in(1).enqueueSeq(inputs(1).map(_.U))
        }.fork {
          dut.io.in(2).enqueueSeq(inputs(2).map(_.U))
        }.fork {
          dut.io.in(3).enqueueSeq(inputs(3).map(_.U))
        }.fork {
          dut.io.in(4).enqueueSeq(inputs(4).map(_.U))
        }.fork {
          dut.io.out.expectDequeueSeq(expected.map(_.U))
        }.join()
      }
    }
  }

}
