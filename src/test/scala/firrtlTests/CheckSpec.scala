// SPDX-License-Identifier: Apache-2.0

package firrtlTests

import org.scalatest._
import firrtl.{CircuitState, Parser, Transform, UnknownForm}
import firrtl.ir.Circuit
import firrtl.passes.{
  CheckFlows,
  CheckHighForm,
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
  val defaultPasses = Seq(ToWorkingIR, CheckHighForm)
  def checkHighInput(input: String) = {
    defaultPasses.foldLeft(Parser.parse(input.split("\n").toIterator)) { (c: Circuit, p: Pass) =>
      p.run(c)
    }
  }

  "CheckHighForm" should "disallow Chirrtl-style memories" in {
    val input =
      """circuit foo :
        |  module foo :
        |    input clock : Clock
        |    input addr : UInt<2>
        |    smem mem : UInt<1>[4]""".stripMargin
    intercept[CheckHighForm.IllegalChirrtlMemException] {
      checkHighInput(input)
    }
  }

  "Memories with flip in the data type" should "throw an exception" in {
    val input =
      """circuit Unit :
        |  module Unit :
        |    mem m :
        |      data-type => {a : {b : {flip c : UInt<32>}}}
        |      depth => 32
        |      read-latency => 0
        |      write-latency => 1""".stripMargin
    intercept[CheckHighForm.MemWithFlipException] {
      checkHighInput(input)
    }
  }

  "Memories with zero write latency" should "throw an exception" in {
    val passes = Seq(ToWorkingIR, CheckHighForm)
    val input =
      """circuit Unit :
        |  module Unit :
        |    mem m :
        |      data-type => UInt<32>
        |      depth => 32
        |      read-latency => 0
        |      write-latency => 0""".stripMargin
    intercept[CheckHighForm.IllegalMemLatencyException] {
      passes.foldLeft(Parser.parse(input.split("\n").toIterator)) { (c: Circuit, p: Pass) =>
        p.run(c)
      }
    }
  }

  "Registers with flip in the type" should "throw an exception" in {
    val input =
      """circuit Unit :
        |  module Unit :
        |    input clk : Clock
        |    input in : UInt<32>
        |    output out : UInt<32>
        |    reg r : {a : UInt<32>, flip b : UInt<32>}, clk
        |    out <= in""".stripMargin
    intercept[CheckHighForm.RegWithFlipException] {
      checkHighInput(input)
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

  "Instance loops a -> b -> a" should "be detected" in {
    val input =
      """
        |circuit Foo :
        |  module Foo :
        |    input a : UInt<32>
        |    output b : UInt<32>
        |    inst bar of Bar
        |    bar.a <= a
        |    b <= bar.b
        |
        |  module Bar :
        |    input a : UInt<32>
        |    output b : UInt<32>
        |    inst foo of Foo
        |    foo.a <= a
        |    b <= foo.b
      """.stripMargin
    intercept[CheckHighForm.InstanceLoop] {
      checkHighInput(input)
    }
  }

  "Instance loops a -> b -> c -> a" should "be detected" in {
    val input =
      """
        |circuit Dog :
        |  module Dog :
        |    input a : UInt<32>
        |    output b : UInt<32>
        |    inst bar of Cat
        |    bar.a <= a
        |    b <= bar.b
        |
        |  module Cat :
        |    input a : UInt<32>
        |    output b : UInt<32>
        |    inst ik of Ik
        |    ik.a <= a
        |    b <= ik.b
        |
        |  module Ik :
        |    input a : UInt<32>
        |    output b : UInt<32>
        |    inst foo of Dog
        |    foo.a <= a
        |    b <= foo.b
        |      """.stripMargin
    intercept[CheckHighForm.InstanceLoop] {
      checkHighInput(input)
    }
  }

  "Instance loops a -> a" should "be detected" in {
    val input =
      """
        |circuit Apple :
        |  module Apple :
        |    input a : UInt<32>
        |    output b : UInt<32>
        |    inst recurse_foo of Apple
        |    recurse_foo.a <= a
        |    b <= recurse_foo.b
        |      """.stripMargin
    intercept[CheckHighForm.InstanceLoop] {
      checkHighInput(input)
    }
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
      CheckHighForm,
      ResolveKinds,
      InferTypes,
      CheckTypes,
      ResolveFlows,
      CheckFlows,
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
    val passes = Seq(ToWorkingIR, CheckHighForm, ResolveKinds, InferTypes, CheckTypes)
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
    val passes = Seq(ToWorkingIR, CheckHighForm, ResolveKinds, InferTypes, CheckTypes)
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

  behavior.of("Uniqueness")
  for ((description, input) <- CheckSpec.nonUniqueExamples) {
    it should s"be asserted for $description" in {
      assertThrows[CheckHighForm.NotUniqueException] {
        Seq(ToWorkingIR, CheckHighForm).foldLeft(Parser.parse(input)) { case (c, tx) => tx.run(c) }
      }
    }
  }

  s"Duplicate module names" should "throw an exception" in {
    val input =
      s"""|circuit bar :
          |  module bar :
          |    input i : UInt<8>
          |    output o : UInt<8>
          |    o <= i
          |  module dup :
          |    input i : UInt<8>
          |    output o : UInt<8>
          |    o <= i
          |  module dup :
          |    input i : UInt<8>
          |    output o : UInt<8>
          |    o <= not(i)
          |""".stripMargin
    assertThrows[CheckHighForm.ModuleNameNotUniqueException] {
      try {
        checkHighInput(input)
      } catch {
        case e: firrtl.passes.PassExceptions => throw e.exceptions.head
      }
    }
  }

  s"Defnames that conflict with pure-FIRRTL module names" should "throw an exception" in {
    val input =
      s"""|circuit bar :
          |  module bar :
          |    input i : UInt<8>
          |    output o : UInt<8>
          |    o <= i
          |  extmodule dup :
          |    input i : UInt<8>
          |    output o : UInt<8>
          |    defname = bar
          |""".stripMargin
    assertThrows[CheckHighForm.DefnameConflictException] {
      checkHighInput(input)
    }
  }

  "Conditionally statements" should "create a new scope" in {
    val input =
      s"""|circuit scopes:
          |  module scopes:
          |    input i: UInt<1>
          |    output o: UInt<1>
          |    when i:
          |      node x = not(i)
          |    o <= and(x, i)
          |""".stripMargin
    assertThrows[CheckHighForm.UndeclaredReferenceException] {
      checkHighInput(input)
    }
  }

  "Attempting to shadow a component name" should "throw an error" in {
    val input =
      s"""|circuit scopes:
          |  module scopes:
          |    input i: UInt<1>
          |    output o: UInt<1>
          |    wire x: UInt<1>
          |    when i:
          |      node x = not(i)
          |    o <= and(x, i)
          |""".stripMargin
    assertThrows[CheckHighForm.NotUniqueException] {
      checkHighInput(input)
    }
  }

  "Attempting to shadow a statement name" should "throw an error" in {
    val input =
      s"""|circuit scopes:
          |  module scopes:
          |    input c: Clock
          |    input i: UInt<1>
          |    output o: UInt<1>
          |    wire x: UInt<1>
          |    when i:
          |      stop(c, UInt(1), 1) : x
          |    o <= and(x, i)
          |""".stripMargin
    assertThrows[CheckHighForm.NotUniqueException] {
      checkHighInput(input)
    }
  }

  "Colliding statement names" should "throw an error" in {
    val input =
      s"""|circuit test:
          |  module test:
          |    input c: Clock
          |    stop(c, UInt(1), 1) : x
          |    stop(c, UInt(1), 1) : x
          |""".stripMargin
    assertThrows[CheckHighForm.NotUniqueException] {
      checkHighInput(input)
    }
  }

  "Conditionally statements" should "create separate consequent and alternate scopes" in {
    val input =
      s"""|circuit scopes:
          |  module scopes:
          |    input i: UInt<1>
          |    output o: UInt<1>
          |    o <= i
          |    when i:
          |      node x = not(i)
          |    else:
          |      o <= and(x, i)
          |""".stripMargin
    assertThrows[CheckHighForm.UndeclaredReferenceException] {
      checkHighInput(input)
    }
  }

  behavior.of("CheckHighForm running on circuits containing ExtModules")

  it should "throw an exception if parameterless ExtModules have the same ports, but different widths" in {
    val input =
      s"""|circuit Foo:
          |  extmodule Bar:
          |    input a: UInt<1>
          |    defname = bar
          |  extmodule Baz:
          |    input a: UInt<2>
          |    defname = bar
          |  module Foo:
          |    skip
          |""".stripMargin
    assertThrows[CheckHighForm.DefnameDifferentPortsException] {
      try {
        checkHighInput(input)
      } catch {
        case e: firrtl.passes.PassExceptions => throw e.exceptions.head
      }
    }
  }

  it should "throw an exception if ExtModules have different port names, but identical widths" in {
    val input =
      s"""|circuit Foo:
          |  extmodule Bar:
          |    input a: UInt<1>
          |    defname = bar
          |  extmodule Baz:
          |    input b: UInt<1>
          |    defname = bar
          |  module Foo:
          |    skip
          |""".stripMargin
    assertThrows[CheckHighForm.DefnameDifferentPortsException] {
      try {
        checkHighInput(input)
      } catch {
        case e: firrtl.passes.PassExceptions => throw e.exceptions.head
      }
    }
  }

  it should "NOT throw an exception if ExtModules have parameters, matching port names, but different widths" in {
    val input =
      s"""|circuit Foo:
          |  extmodule Bar:
          |    input a: UInt<1>
          |    defname = bar
          |    parameter width = 1
          |  extmodule Baz:
          |    input a: UInt<2>
          |    defname = bar
          |    parameter width = 2
          |  module Foo:
          |    skip
          |""".stripMargin
    checkHighInput(input)
  }

  it should "throw an exception if ExtModules have matching port names and widths, but a different order" in {
    val input =
      s"""|circuit Foo:
          |  extmodule Bar:
          |    input a: UInt<1>
          |    input b: UInt<1>
          |    defname = bar
          |  extmodule Baz:
          |    input b: UInt<1>
          |    input a: UInt<1>
          |    defname = bar
          |  module Foo:
          |    skip
          |""".stripMargin
    assertThrows[CheckHighForm.DefnameDifferentPortsException] {
      try {
        checkHighInput(input)
      } catch {
        case e: firrtl.passes.PassExceptions => throw e.exceptions.head
      }
    }
  }

  it should "throw an exception if ExtModules have matching port names, but one is a Clock and one is a UInt<1>" in {
    val input =
      s"""|circuit Foo:
          |  extmodule Bar:
          |    input a: UInt<1>
          |    defname = bar
          |  extmodule Baz:
          |    input a: Clock
          |    defname = bar
          |  module Foo:
          |    skip
          |""".stripMargin
    assertThrows[CheckHighForm.DefnameDifferentPortsException] {
      try {
        checkHighInput(input)
      } catch {
        case e: firrtl.passes.PassExceptions => throw e.exceptions.head
      }
    }
  }

  it should "throw an exception if ExtModules have differing concrete reset types" in {
    def input(rst1: String, rst2: String) =
      s"""|circuit Foo:
          |  extmodule Bar:
          |    input rst: $rst1
          |    defname = bar
          |  extmodule Baz:
          |    input rst: $rst2
          |    defname = bar
          |  module Foo:
          |    skip
          |""".stripMargin
    info("exception thrown for 'UInt<1>' compared to 'AsyncReset'")
    assertThrows[CheckHighForm.DefnameDifferentPortsException] {
      try {
        checkHighInput(input("UInt<1>", "AsyncReset"))
      } catch {
        case e: firrtl.passes.PassExceptions => throw e.exceptions.head
      }
    }
    info("exception thrown for 'UInt<1>' compared to 'Reset'")
    assertThrows[CheckHighForm.DefnameDifferentPortsException] {
      try {
        checkHighInput(input("UInt<1>", "Reset"))
      } catch {
        case e: firrtl.passes.PassExceptions => throw e.exceptions.head
      }
    }
  }

  it should "throw an exception if a statement name is used as a reference" in {
    val src = """
                |circuit test:
                |  module test:
                |    input clock: Clock
                |    output a: UInt<2>
                |    stop(clock, UInt(1), 1) : hello
                |    a <= hello
                |""".stripMargin
    assertThrows[CheckHighForm.UndeclaredReferenceException] {
      checkHighInput(src)
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
