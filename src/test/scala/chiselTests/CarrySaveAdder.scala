// SPDX-License-Identifier: Apache-2.0

package chiselTests

import scala.util.Random

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
import chisel3.util.{Counter, Csa, PopCount}
import chisel3.util.random.LFSR
import org.scalatest.propspec.AnyPropSpec
import org.scalatest.matchers.should.Matchers
import circt.stage.ChiselStage

class CsaTester(termWidths: Seq[Int], boolCount: Int) extends Module {

  // Cannot exhaustively simulate all input combinations.
  // Instead: 1- test correctness around min-and-max input values
  //          2- test correctness for random input values

  val (_, expired) = Counter(0 to 200)
  when(expired) { stop() }

  // Directed test. Each term starts at zero. Decrement one term at a time, round-robin.
  val (termDecrementPtr, _) = Counter(0 until termWidths.length + boolCount)
  val termsCounting = termWidths.zipWithIndex.map { case (tW, idx) =>
    val term = RegInit(0.U(tW.W))
    when(idx.U === termDecrementPtr) { term := term - 1.U }
    term
  }
  val bools = Seq.fill(boolCount)(RegInit(false.B))
  bools.zipWithIndex.foreach { case (b, idx) =>
    when(idx.U === termDecrementPtr - termWidths.length.U) { b := ~b }
  }

  // Random test. LFSR does not work for bitwidths 0 and 1
  val termsRandom = termWidths.map { tW => if (tW >= 2) LFSR(tW) else tW.U }
  val testCases = Seq(termsCounting, termsRandom)

  testCases.foreach { csaInput => // parallel testing circuitry for both tests
    val csaResult = Csa(csaInput, bools)
    val (sum, cry) = Csa.sumCarry(csaInput, bools)
    val refResult = csaInput.reduce((a, b) => a +& b) +& PopCount(bools)
    assert(csaInput.forall(_.isWidthKnown), "Testcase error: should know the width of input terms")
    assert(
      csaResult.getWidth <= refResult.getWidth,
      s"csaResult width ${csaResult.getWidth} should not exceed refResult width ${refResult.getWidth}\n"
    )
    assert(csaResult === refResult, s"Wrong result of CSA final sum, $csaInput")
    assert((sum +& cry) === refResult, s"Wrong result of CSA output in redundant form, $csaInput")
  }
}

class CsaSpec extends AnyPropSpec with PropertyUtils with ChiselSim with Matchers with LogUtils {
  property(s"Carry-Save Adder (10 inputs, 20-bit-wide each + some bools) should return correct result") {
    simulate(new CsaTester(Seq.fill(10)(20), 5))(RunUntilFinished(1000))
  }

  val prng = new Random(seed = 1234567)
  for (n <- ((1 to 5) ++ (10 to 25 by 5))) { // number of CSA input terms
    val testCsaTermWidths = prng.shuffle(Seq.range(0, 31)).take(n) // constrained random width of each CSA input term
    property(s"Carry-Save Adder with $n input terms + ${n / 3} bools should return correct result") {
      simulate(new CsaTester(testCsaTermWidths, n / 3))(RunUntilFinished(1000))
    }
  }

  property("Carry-Save Adder should warn about unknown-width input terms") {
    val (log, _) = grabLog(ChiselStage.emitCHIRRTL(new RawModule {
      val myUInts = Seq.fill(4)(Wire(UInt()))
      Csa(myUInts)
    }))
    log should include("Cannot optimize width of carry vector because width of input term ")
  }
}
