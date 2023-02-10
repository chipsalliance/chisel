// SPDX-License-Identifier: Apache-2.0

package firrtlTests

import org.scalatest._
import firrtl.{CircuitState, Parser, Transform, UnknownForm}
import firrtl.ir.Circuit
import firrtl.passes.{
  CheckTypes,
  CheckWidths,
  InferTypes,
  InferWidths,
  Pass,
  PassException,
  ResolveFlows,
  ResolveKinds,
  ToWorkingIR
}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class CheckSpec extends AnyFlatSpec with Matchers {
  val defaultPasses = Seq(ToWorkingIR)
  def checkHighInput(input: String) = {
    defaultPasses.foldLeft(Parser.parse(input.split("\n").toIterator)) { (c: Circuit, p: Pass) =>
      p.run(c)
    }
  }

  behavior.of("Check Types")

  def runCheckTypes(input: String) = {
    val passes = List(InferTypes, CheckTypes)
    val wrapped = "circuit test:\n  module test:\n    " + input.replaceAll("\n", "\n    ")
    passes.foldLeft(Parser.parse(wrapped)) { case (c, p) => p.run(c) }
  }

  it should "disallow mux enable conditions that are not 1-bit UInts (or unknown width)" in {
    def mk(tpe: String) =
      s"""|input en : $tpe
          |input foo : UInt<8>
          |input bar : UInt<8>
          |node x = mux(en, foo, bar)""".stripMargin
    a[CheckTypes.MuxCondUInt] shouldBe thrownBy { runCheckTypes(mk("SInt<1>")) }
    a[CheckTypes.MuxCondUInt] shouldBe thrownBy { runCheckTypes(mk("SInt")) }
    a[CheckTypes.MuxCondUInt] shouldBe thrownBy { runCheckTypes(mk("UInt<3>")) }
    a[CheckTypes.MuxCondUInt] shouldBe thrownBy { runCheckTypes(mk("Clock")) }
    a[CheckTypes.MuxCondUInt] shouldBe thrownBy { runCheckTypes(mk("AsyncReset")) }
    runCheckTypes(mk("UInt"))
    runCheckTypes(mk("UInt<1>"))
  }

  it should "disallow when predicates that are not 1-bit UInts (or unknown width)" in {
    def mk(tpe: String) =
      s"""|input en : $tpe
          |input foo : UInt<8>
          |input bar : UInt<8>
          |output out : UInt<8>
          |when en :
          |  out <= foo
          |else:
          |  out <= bar""".stripMargin
    a[CheckTypes.PredNotUInt] shouldBe thrownBy { runCheckTypes(mk("SInt<1>")) }
    a[CheckTypes.PredNotUInt] shouldBe thrownBy { runCheckTypes(mk("SInt")) }
    a[CheckTypes.PredNotUInt] shouldBe thrownBy { runCheckTypes(mk("UInt<3>")) }
    a[CheckTypes.PredNotUInt] shouldBe thrownBy { runCheckTypes(mk("Clock")) }
    a[CheckTypes.PredNotUInt] shouldBe thrownBy { runCheckTypes(mk("AsyncReset")) }
    runCheckTypes(mk("UInt"))
    runCheckTypes(mk("UInt<1>"))
  }

  it should "disallow print enables that are not 1-bit UInts (or unknown width)" in {
    def mk(tpe: String) =
      s"""|input en : $tpe
          |input clock : Clock
          |printf(clock, en, "Hello World!\\n")""".stripMargin
    a[CheckTypes.EnNotUInt] shouldBe thrownBy { runCheckTypes(mk("SInt<1>")) }
    a[CheckTypes.EnNotUInt] shouldBe thrownBy { runCheckTypes(mk("SInt")) }
    a[CheckTypes.EnNotUInt] shouldBe thrownBy { runCheckTypes(mk("UInt<3>")) }
    a[CheckTypes.EnNotUInt] shouldBe thrownBy { runCheckTypes(mk("Clock")) }
    a[CheckTypes.EnNotUInt] shouldBe thrownBy { runCheckTypes(mk("AsyncReset")) }
    runCheckTypes(mk("UInt"))
    runCheckTypes(mk("UInt<1>"))
  }

  it should "disallow stop enables that are not 1-bit UInts (or unknown width)" in {
    def mk(tpe: String) =
      s"""|input en : $tpe
          |input clock : Clock
          |stop(clock, en, 0)""".stripMargin
    a[CheckTypes.EnNotUInt] shouldBe thrownBy { runCheckTypes(mk("SInt<1>")) }
    a[CheckTypes.EnNotUInt] shouldBe thrownBy { runCheckTypes(mk("SInt")) }
    a[CheckTypes.EnNotUInt] shouldBe thrownBy { runCheckTypes(mk("UInt<3>")) }
    a[CheckTypes.EnNotUInt] shouldBe thrownBy { runCheckTypes(mk("Clock")) }
    a[CheckTypes.EnNotUInt] shouldBe thrownBy { runCheckTypes(mk("AsyncReset")) }
    runCheckTypes(mk("UInt"))
    runCheckTypes(mk("UInt<1>"))
  }

  it should "disallow verif node predicates that are not 1-bit UInts (or unknown width)" in {
    def mk(tpe: String) =
      s"""|input en : $tpe
          |input cond : UInt<1>
          |input clock : Clock
          |assert(clock, en, cond, "Howdy!")""".stripMargin
    a[CheckTypes.PredNotUInt] shouldBe thrownBy { runCheckTypes(mk("SInt<1>")) }
    a[CheckTypes.PredNotUInt] shouldBe thrownBy { runCheckTypes(mk("SInt")) }
    a[CheckTypes.PredNotUInt] shouldBe thrownBy { runCheckTypes(mk("UInt<3>")) }
    a[CheckTypes.PredNotUInt] shouldBe thrownBy { runCheckTypes(mk("Clock")) }
    a[CheckTypes.PredNotUInt] shouldBe thrownBy { runCheckTypes(mk("AsyncReset")) }
    runCheckTypes(mk("UInt"))
    runCheckTypes(mk("UInt<1>"))
  }

  it should "disallow verif node enables that are not 1-bit UInts (or unknown width)" in {
    def mk(tpe: String) =
      s"""|input en : UInt<1>
          |input cond : $tpe
          |input clock : Clock
          |assert(clock, en, cond, "Howdy!")""".stripMargin
    a[CheckTypes.EnNotUInt] shouldBe thrownBy { runCheckTypes(mk("SInt<1>")) }
    a[CheckTypes.EnNotUInt] shouldBe thrownBy { runCheckTypes(mk("SInt")) }
    a[CheckTypes.EnNotUInt] shouldBe thrownBy { runCheckTypes(mk("UInt<3>")) }
    a[CheckTypes.EnNotUInt] shouldBe thrownBy { runCheckTypes(mk("Clock")) }
    a[CheckTypes.EnNotUInt] shouldBe thrownBy { runCheckTypes(mk("AsyncReset")) }
    runCheckTypes(mk("UInt"))
    runCheckTypes(mk("UInt<1>"))
  }

  "Instance loops should not have false positives" should "be detected" in {
    val input =
      """
        |circuit Hammer :
        |  module Hammer :
        |    input a : UInt<32>
        |    output b : UInt<32>
        |    inst bar of Chisel
        |    bar.a <= a
        |    b <= bar.b
        |
        |  module Chisel :
        |    input a : UInt<32>
        |    output b : UInt<32>
        |    inst ik of Saw
        |    ik.a <= a
        |    b <= ik.b
        |
        |  module Saw :
        |    input a : UInt<32>
        |    output b : UInt<32>
        |    b <= a
        |      """.stripMargin
    checkHighInput(input)
  }

  "Clock Types" should "be connectable" in {
    val passes = Seq(
      ToWorkingIR,
      ResolveKinds,
      InferTypes,
      CheckTypes,
      ResolveFlows,
      new InferWidths,
      CheckWidths
    )
    val input =
      """
        |circuit TheRealTop :
        |
        |  module Top :
        |    output io : {flip debug_clk : Clock}
        |
        |  extmodule BlackBoxTop :
        |    input jtag : {TCK : Clock}
        |
        |  module TheRealTop :
        |    input clock : Clock
        |    input reset : UInt<1>
        |    output io : {flip jtag : {TCK : Clock}}
        |
        |    io is invalid
        |    inst sub of Top
        |    sub.io is invalid
        |    inst bb of BlackBoxTop
        |    bb.jtag is invalid
        |    bb.jtag <- io.jtag
        |
        |    sub.io.debug_clk <= io.jtag.TCK
        |
        |""".stripMargin
    passes.foldLeft(CircuitState(Parser.parse(input.split("\n").toIterator), UnknownForm)) {
      (c: CircuitState, p: Transform) => p.runTransform(c)
    }
  }

  "Clocks with types other than ClockType" should "throw an exception" in {
    val passes = Seq(ToWorkingIR, ResolveKinds, InferTypes, CheckTypes)
    val input =
      """
        |circuit Top :
        |
        |  module Top :
        |    input clk : UInt<1>
        |    input i : UInt<1>
        |    output o : UInt<1>
        |
        |    reg r : UInt<1>, clk
        |    r <= i
        |    o <= r
        |
        |""".stripMargin
    intercept[CheckTypes.RegReqClk] {
      passes.foldLeft(Parser.parse(input.split("\n").toIterator)) { (c: Circuit, p: Pass) =>
        p.run(c)
      }
    }
  }

  "Illegal reset type" should "throw an exception" in {
    val passes = Seq(ToWorkingIR, ResolveKinds, InferTypes, CheckTypes)
    val input =
      """
        |circuit Top :
        |
        |  module Top :
        |    input clk : Clock
        |    input reset : UInt<2>
        |    input i : UInt<1>
        |    output o : UInt<1>
        |
        |    reg r : UInt<1>, clk with : (reset => (reset, UInt<1>("h00")))
        |    r <= i
        |    o <= r
        |
        |""".stripMargin
    intercept[CheckTypes.IllegalResetType] {
      passes.foldLeft(Parser.parse(input.split("\n").toIterator)) { (c: Circuit, p: Pass) =>
        p.run(c)
      }
    }
  }

  for (op <- List("shl", "shr", "pad", "head", "tail", "incp", "decp")) {
    s"$op by negative amount" should "result in an error" in {
      val amount = -1
      val input =
        s"""circuit Unit :
           |  module Unit :
           |    input x: UInt<3>
           |    output z: UInt
           |    z <= $op(x, $amount)""".stripMargin
      val exception = intercept[PassException] {
        checkHighInput(input)
      }
      exception.getMessage should include(s"Primop $op argument $amount < 0")
    }
  }

  // Check negative bits constant, too
  for (args <- List((3, 4), (0, -1))) {
    val opExp = s"bits(in, ${args._1}, ${args._2})"
    s"Illegal bit extract ${opExp}" should "throw an exception" in {
      val input =
        s"""|circuit bar :
            |  module bar :
            |    input in : UInt<8>
            |    output foo : UInt
            |    foo <= ${opExp}""".stripMargin
      val exception = intercept[PassException] {
        checkHighInput(input)
      }
    }
  }

}

object CheckSpec {
  val nonUniqueExamples = List(
    ("two ports with the same name", """|circuit Top:
                                       |  module Top:
                                       |    input a: UInt<1>
                                       |    input a: UInt<1>""".stripMargin),
    ("two nodes with the same name", """|circuit Top:
                                       |  module Top:
                                       |    node a = UInt<1>("h0")
                                       |    node a = UInt<1>("h0")""".stripMargin),
    ("a port and a node with the same name", """|circuit Top:
                                               |  module Top:
                                               |    input a: UInt<1>
                                               |    node a = UInt<1>("h0") """.stripMargin)
  )
}
