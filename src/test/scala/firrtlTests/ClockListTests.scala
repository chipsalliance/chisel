// See LICENSE for license details.

package firrtlTests

import java.io._
import firrtl._
import firrtl.ir.Circuit
import firrtl.passes._
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
    (writer.toString) should be (check)
  }
  "A->B->C, and A.clock == C.clock" should "still emit C.clock origin" in {
    val input =
      """circuit A :
        |  module A :
        |    input clock: Clock
        |    input clkB: Clock
        |    inst b of B
        |    b.clock <= clkB
        |    b.clkC <= clock
        |  module B :
        |    input clock: Clock
        |    input clkC: Clock
        |    inst c of C
        |    c.clock <= clkC
        |  module C :
        |    input clock: Clock
        |    reg r: UInt<5>, clock
        |""".stripMargin
    val check = 
  """Sourcelist: List(clock, clkB) 
    |Good Origin of clock is clock
    |Good Origin of b.clock is clkB
    |Good Origin of b$c.clock is clock
    |""".stripMargin
    val c = passes.foldLeft(parse(input)) {
      (c: Circuit, p: Pass) => p.run(c)
    }
    val writer = new StringWriter()
    val retC = new ClockList("A", writer).run(c)
    (writer.toString) should be (check)
  }
  "Have not circuit main be top of clocklist pass" should "still work" in {
    val input =
      """circuit A :
        |  module A :
        |    input clock: Clock
        |    input clkB: Clock
        |    inst b of B
        |    inst d of D
        |    b.clock <= clkB
        |    b.clkC <= clock
        |  module B :
        |    input clock: Clock
        |    input clkC: Clock
        |    inst c of C
        |    c.clock <= clkC
        |  module C :
        |    input clock: Clock
        |    reg r: UInt<5>, clock
        |  extmodule D :
        |    input clock: Clock
        |""".stripMargin
    val check =
  """Sourcelist: List(clock, clkC) 
    |Good Origin of clock is clock
    |Good Origin of c.clock is clkC
    |""".stripMargin
    val c = passes.foldLeft(parse(input)) {
      (c: Circuit, p: Pass) => p.run(c)
    }
    val writer = new StringWriter()
    val retC = new ClockList("B", writer).run(c)
    (writer.toString) should be (check)
  }
}
