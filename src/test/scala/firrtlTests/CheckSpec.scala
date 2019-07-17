// See LICENSE for license details.

package firrtlTests

import org.scalatest._
import firrtl.{Parser, CircuitState, UnknownForm, Transform}
import firrtl.ir.Circuit
import firrtl.passes.{Pass,ToWorkingIR,CheckHighForm,ResolveKinds,InferTypes,CheckTypes,PassException,InferWidths,CheckWidths,ResolveFlows,CheckFlows}

class CheckSpec extends FlatSpec with Matchers {
  val defaultPasses = Seq(ToWorkingIR, CheckHighForm)
  def checkHighInput(input: String) = {
    defaultPasses.foldLeft(Parser.parse(input.split("\n").toIterator)) {
      (c: Circuit, p: Pass) => p.run(c)
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
    val passes = Seq(
      ToWorkingIR,
      CheckHighForm)
    val input =
      """circuit Unit :
        |  module Unit :
        |    mem m :
        |      data-type => UInt<32>
        |      depth => 32
        |      read-latency => 0
        |      write-latency => 0""".stripMargin
    intercept[CheckHighForm.IllegalMemLatencyException] {
      passes.foldLeft(Parser.parse(input.split("\n").toIterator)) {
        (c: Circuit, p: Pass) => p.run(c)
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
      CheckWidths)
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
    val passes = Seq(
      ToWorkingIR,
      CheckHighForm,
      ResolveKinds,
      InferTypes,
      CheckTypes)
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
      passes.foldLeft(Parser.parse(input.split("\n").toIterator)) {
        (c: Circuit, p: Pass) => p.run(c)
      }
    }
  }

  "Illegal reset type" should "throw an exception" in {
    val passes = Seq(
      ToWorkingIR,
      CheckHighForm,
      ResolveKinds,
      InferTypes,
      CheckTypes)
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
      passes.foldLeft(Parser.parse(input.split("\n").toIterator)) {
        (c: Circuit, p: Pass) => p.run(c)
      }
    }
  }

  for (op <- List("shl", "shr")) {
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
      exception.getMessage should include (s"Primop $op argument $amount < 0")
    }
  }

  "LSB larger than MSB in bits" should "throw an exception" in {
    val input =
      """|circuit bar :
         |  module bar :
         |    input in : UInt<8>
         |    output foo : UInt
         |    foo <= bits(in, 3, 4)
         |      """.stripMargin
    val exception = intercept[PassException] {
      checkHighInput(input)
    }
  }

  behavior of "Uniqueness"
  for ((description, input) <- CheckSpec.nonUniqueExamples) {
    it should s"be asserted for $description" in {
      assertThrows[CheckHighForm.NotUniqueException] {
        Seq(ToWorkingIR, CheckHighForm).foldLeft(Parser.parse(input)){ case (c, tx) => tx.run(c) }
      }
    }
  }
}

object CheckSpec {
  val nonUniqueExamples = List(
    ("two ports with the same name",
     """|circuit Top:
        |  module Top:
        |    input a: UInt<1>
        |    input a: UInt<1>""".stripMargin),
    ("two nodes with the same name",
     """|circuit Top:
        |  module Top:
        |    node a = UInt<1>("h0")
        |    node a = UInt<1>("h0")""".stripMargin),
    ("a port and a node with the same name",
     """|circuit Top:
        |  module Top:
        |    input a: UInt<1>
        |    node a = UInt<1>("h0") """.stripMargin) )
  }
