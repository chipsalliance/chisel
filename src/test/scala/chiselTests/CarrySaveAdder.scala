// SPDX-License-Identifier: Apache-2.0

package chiselTests

import scala.util.Random

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
import chisel3.util.{Counter, Csa}
import chisel3.util.random.LFSR
import org.scalatest.propspec.AnyPropSpec
import org.scalatest.matchers.should.Matchers

class CsaTester(termWidths: Seq[Int]) extends Module {

  // Cannot exhaustively simulate all input combinations.
  // Instead: 1- test correctness around min-and-max input values
  //          2- test correctness for random input values

  val (_, expired) = Counter(0 to 200)
  when(expired) { stop() }

  // Directed test. Each term starts at zero. Decrement one term at a time, round-robin.
  val (termDecrementPtr, _) = Counter(0 until termWidths.length)
  val termsCounting = termWidths.zipWithIndex.map { case (tW, idx) =>
    val term = RegInit(0.U(tW.W))
    when(idx.U === termDecrementPtr) { term := term - 1.U }
    term
  }

  // Random test. LFSR does not work for bitwidths 0 and 1
  val termsRandom = termWidths.map { tW => if (tW >= 2) LFSR(tW) else tW.U }
  val testCases = Seq(termsCounting, termsRandom)

  testCases.foreach { csaInput => // parallel testing circuitry foreach
    val csaOutput = Csa(csaInput)
    assert(csaOutput.length <= 2, s"CSA tree has more than 2 output terms")
    val csaResult = csaOutput.reduce((a, b) => a +& b)
    val refResult = csaInput.reduce((a, b) => a +& b)
    assert(csaResult === refResult, s"Wrong result at CSA output, $csaInput")
  }
}

class CsaSpec extends AnyPropSpec with PropertyUtils with ChiselSim {
  property(s"CSA adder reduction tree (10 inputs, 20-bit-wide each) should return the correct result") {
    simulate(new CsaTester(Seq.fill(10)(20)))(RunUntilFinished(1000))
  }

  val prng = new Random(seed = 1234567)
  for (n <- ((1 to 5) ++ (10 to 25 by 5))) { // number of CSA input terms
    val testCsaTermWidths = prng.shuffle(Seq.range(0, 31)).take(n) // constrained random width of each CSA input term
    property(s"CSA adder reduction tree with $n input terms of different widths should return the correct result") {
      simulate(new CsaTester(testCsaTermWidths))(RunUntilFinished(1000))
    }
  }
}
