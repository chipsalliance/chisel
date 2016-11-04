// See LICENSE for license details.

package firrtlTests

import java.io._
import org.scalatest._
import org.scalatest.prop._
import firrtl.Parser
import firrtl.ir.Circuit
import firrtl.passes._

class ChirrtlSpec extends FirrtlFlatSpec {
  def passes = Seq(
    CheckChirrtl,
    CInferTypes,
    CInferMDir,
    RemoveCHIRRTL,
    ToWorkingIR,            
    CheckHighForm,
    ResolveKinds,
    InferTypes,
    CheckTypes,
    ResolveGenders,
    CheckGenders,
    InferWidths,
    CheckWidths,
    PullMuxes,
    ExpandConnects,
    RemoveAccesses,
    ExpandWhens,
    CheckInitialization
  )

  "Chirrtl memories" should "allow ports with clocks defined after the memory" in {
    val input =
     """circuit Unit :
       |  module Unit :
       |    input clk : Clock
       |    smem ram : UInt<32>[128]
       |    node newClock = clk
       |    infer mport x = ram[UInt(2)], newClock
       |    x <= UInt(3)
       |    when UInt(1) :
       |      infer mport y = ram[UInt(4)], newClock
       |      y <= UInt(5)
       """.stripMargin
    passes.foldLeft(Parser.parse(input.split("\n").toIterator)) {
      (c: Circuit, p: Pass) => p.run(c)
    }
  }

  "Chirrtl" should "catch undeclared wires" in {
    val input =
     """circuit Unit :
       |  module Unit :
       |    input clk : Clock
       |    smem ram : UInt<32>[128]
       |    node newClock = clk
       |    infer mport x = ram[UInt(2)], newClock
       |    x <= UInt(3)
       |    when UInt(1) :
       |      infer mport y = ram[UInt(4)], newClock
       |      y <= z
       """.stripMargin
    intercept[PassException] {
      passes.foldLeft(Parser.parse(input.split("\n").toIterator)) {
        (c: Circuit, p: Pass) => p.run(c)
      }
    }
  }
}

class ChirrtlMemsExecutionTest extends ExecutionTest("ChirrtlMems", "/features")

