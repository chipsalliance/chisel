package chiselTests.simulator

import chisel3._
import chisel3.util._
import chisel3.simulator._

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.must.Matchers

class ThreadedChiselSimSpec extends AnyFunSpec with ChiselSimTester {

  val rand = scala.util.Random

  describe("ThreadedChiselSim") {
    it("should work for a FIFO") {

      val numTests = 133

      val w = 32
      val fifoDepth = 7

      test(
        new Queue(UInt(w.W), fifoDepth, pipe = true, flow = false),
        ChiselSimSettings.verilatorBackend(resetWorkspace = true, traceStyle = TraceStyle.Vcd())
      ) { dut =>
        for (testCase <- 1 to numTests) {

          val inputs = Seq.fill(rand.between(0, 11 * dut.entries))(BigInt(w, rand))
          val expected = inputs

          // println(s"testCase #${testCase}/${numTests}")

          dut.io.deq.valid.expect(false.B)
          dut.io.enq.ready.expect(true.B)

          fork {
            dut.io.enq.enqueueSeq(inputs.map(_.U))
            for (_ <- 0 until rand.between(0, 7)) {
              dut.clock.step()
            }
          }.fork {
            dut.io.deq.expectDequeueSeq(expected.map(_.U))
            for (_ <- 0 until rand.between(0, 11)) {
              dut.io.deq.valid.expect(false.B)
              dut.io.count.expect(0)
              dut.clock.step()
            }
          }.joinAndStep()
        }
      }
    }
  }

}

class ThreadedChiselSimSpec2 extends AnyFlatSpec with ChiselSimTester {

  val rand = scala.util.Random

  it should "work for multiple consumer threads" in {

    val numTests = 133

    test(
      new SplitMod(4, 32),
      ChiselSimSettings.verilatorBackend(resetWorkspace = true, traceStyle = TraceStyle.Vcd())
    ) { dut =>
      for (testCase <- 1 to numTests) {
        val inputs = Seq.fill(rand.between(0, 300))(BigInt(dut.io.in.bits.getWidth, rand))
        val expected = inputs.groupBy(_ % dut.n)

        def checkOut(i: Int) = {
          dut.io.out(i).expectDequeueSeq(expected.getOrElse(i, Seq.empty).map(_.U))
          for (_ <- 0 until rand.between(0, 3)) {
            dut.io.out(i).valid.expect(false.B)
            dut.clock.step()
          }
        }

        fork {
          dut.io.in.enqueueSeq(inputs.map(_.U))
          for (_ <- 0 until rand.between(0, 3)) {
            dut.clock.step()
          }
        }.fork {
          checkOut(0)
        }.fork {
          checkOut(1)
        }.fork {
          checkOut(2)
        }.fork {
          checkOut(3)
        }.join()
      }
    }
  }
}

class ThreadedChiselSimSpec3 extends AnyFunSpec with ChiselSimTester {

  val rand = scala.util.Random

  describe("ThreadedChiselSim") {
    it("should work for a FIFO") {

      val numTests = 133

      val w = 32
      val fifoDepth = 7

      test(
        new Queue(UInt(w.W), fifoDepth, pipe = true, flow = false),
        ChiselSimSettings.verilatorBackend(resetWorkspace = true, traceStyle = TraceStyle.Vcd())
      ) { dut =>
        for (testCase <- 1 to numTests) {

          val inputs = Seq.fill(rand.between(0, 11 * dut.entries))(BigInt(w, rand))
          val expected = inputs

          // println(s"testCase #${testCase}/${numTests}")

          dut.io.deq.valid.expect(false.B)
          dut.io.enq.ready.expect(true.B)

          fork {
            dut.io.enq.enqueueSeq(inputs.map(_.U))
            for (_ <- 0 until rand.between(0, 7)) {
              dut.clock.step()
            }
          }.fork {
            dut.io.deq.expectDequeueSeq(expected.map(_.U))
            for (_ <- 0 until rand.between(0, 11)) {
              dut.io.deq.valid.expect(false.B)
              dut.io.count.expect(0)
              dut.clock.step()
            }
          }.joinAndStep()
        }
      }
    }
  }

  it("should work for multiple consumer threads") {

    val numTests = 133

    test(
      new SplitMod(4, 32),
      ChiselSimSettings.verilatorBackend(resetWorkspace = true, traceStyle = TraceStyle.Vcd())
    ) { dut =>
      for (testCase <- 1 to numTests) {
        val inputs = Seq.fill(rand.between(0, 300))(BigInt(dut.io.in.bits.getWidth, rand))
        val expected = inputs.groupBy(_ % dut.n)

        def checkOut(i: Int) = {
          dut.io.out(i).expectDequeueSeq(expected.getOrElse(i, Seq.empty).map(_.U))
          for (_ <- 0 until rand.between(0, 3)) {
            dut.io.out(i).valid.expect(false.B)
            dut.clock.step()
          }
        }

        fork {
          dut.io.in.enqueueSeq(inputs.map(_.U))
          for (_ <- 0 until rand.between(0, 3)) {
            dut.clock.step()
          }
        }.fork {
          checkOut(0)
        }.fork {
          checkOut(1)
        }.fork {
          checkOut(2)
        }.fork {
          checkOut(3)
        }.join()
      }
    }
  }
}
