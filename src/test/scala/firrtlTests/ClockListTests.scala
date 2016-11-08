package firrtlTests

import java.io._
import org.scalatest._
import org.scalatest.prop._
import firrtl._
import firrtl.ir.Circuit
import firrtl.passes._
import firrtl.Parser.IgnoreInfo
import Annotations._
import clocklist._

class ClockListTests extends FirrtlFlatSpec {
  private def executeTest(input: String, expected: Seq[String], passes: Seq[Pass]) = {
    val c = passes.foldLeft(Parser.parse(input.split("\n").toIterator)) {
      (c: Circuit, p: Pass) => p.run(c)
    }
    val lines = c.serialize.split("\n") map normalized

    expected foreach { e =>
      lines should contain(e)
    }
  }

  def passes = Seq(
    ToWorkingIR,
    ResolveKinds,
    InferTypes,
    ResolveGenders,
    InferWidths
  )

  "Getting clock list" should "work" in {
    val input =
      """circuit Top :
        |  module Top :
        |    input clock: Clock
        |    inst ht of HTop
        |    ht.clock <= clock
        |  module HTop :
        |    input clock: Clock
        |    inst h of Hurricane
        |    h.clkTop <= clock
        |    h.clock <= h.clk1
        |  module Hurricane :
        |    input clock: Clock
        |    input clkTop: Clock
        |    output clk1: Clock
        |    inst b of B
        |    inst c of C
        |    inst clkGen of ClockGen
        |    clkGen.clkTop <= clkTop
        |    clk1 <= clkGen.clk1
        |    b.clock <= clkGen.clk2
        |    c.clock <= clkGen.clk3
        |  module B :
        |    input clock: Clock
        |    reg r: UInt<5>, clock
        |    inst d of D
        |    d.clock <= clock
        |  module C :
        |    input clock: Clock
        |    reg r: UInt<5>, clock
        |  module D :
        |    input clock: Clock
        |    reg r: UInt<5>, clock
        |  extmodule ClockGen :
        |    input clkTop: Clock
        |    output clk1: Clock
        |    output clk2: Clock
        |    output clk3: Clock
        |""".stripMargin
    val check = 
  """Sourcelist: List(h$clkGen$clk1, h$clkGen$clk2, h$clkGen$clk3, clock) 
    |Good Origin of clock is clock
    |Good Origin of h.clock is h$clkGen.clk1
    |Good Origin of h$b.clock is h$clkGen.clk2
    |Good Origin of h$c.clock is h$clkGen.clk3
    |""".stripMargin
    val c = passes.foldLeft(parse(input)) {
      (c: Circuit, p: Pass) => p.run(c)
    }
    val writer = new StringWriter()
    val retC = new ClockList("HTop", writer).run(c)
    (writer.toString()) should be (check)
  }
}
