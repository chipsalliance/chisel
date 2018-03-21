package firrtlTests
package transforms

import firrtl.annotations.{CircuitName, ComponentName, ModuleName}
import firrtl.transforms.{GroupAnnotation, GroupComponents}

class GroupComponentsSpec extends LowTransformSpec {
  def transform = new GroupComponents()
  val top = "Top"
  def topComp(name: String): ComponentName = ComponentName(name, ModuleName(top, CircuitName(top)))
  "The register r" should "be grouped" in {
    val input =
    s"""circuit $top :
        |  module $top :
        |    input clk: Clock
        |    input data: UInt<16>
        |    output out: UInt<16>
        |    reg r: UInt<16>, clk
        |    r <= data
        |    out <= r
      """.stripMargin
    val groups = Seq(
      GroupAnnotation(Seq(topComp("r")), "MyReg", "rInst", Some("_OUT"), Some("_IN"))
    )
    val check =
     s"""circuit Top :
        |  module $top :
        |    input clk: Clock
        |    input data: UInt<16>
        |    output out: UInt<16>
        |    inst rInst of MyReg
        |    rInst.clk_IN <= clk
        |    out <= rInst.r_OUT
        |    rInst.data_IN <= data
        |  module MyReg :
        |    input clk_IN: Clock
        |    output r_OUT: UInt<16>
        |    input data_IN: UInt<16>
        |    reg r: UInt<16>, clk_IN
        |    r_OUT <= r
        |    r <= data_IN
      """.stripMargin
    execute(input, check, groups)
  }

  "The two sets of instances" should "be grouped" in {
    val input =
      s"""circuit $top :
         |  module $top :
         |    output out: UInt<16>
         |    inst c1a of Const1A
         |    inst c2a of Const2A
         |    inst c1b of Const1B
         |    inst c2b of Const2B
         |    node asum = add(c1a.out, c2a.out)
         |    node bsum = add(c1b.out, c2b.out)
         |    out <= add(asum, bsum)
         |  module Const1A :
         |    output out: UInt<8>
         |    out <= UInt(1)
         |  module Const2A :
         |    output out: UInt<8>
         |    out <= UInt(2)
         |  module Const1B :
         |    output out: UInt<8>
         |    out <= UInt(1)
         |  module Const2B :
         |    output out: UInt<8>
         |    out <= UInt(2)
      """.stripMargin
    val groups = Seq(
      GroupAnnotation(Seq(topComp("c1a"), topComp("c2a")/*, topComp("asum")*/), "A", "cA", Some("_OUT"), Some("_IN")),
      GroupAnnotation(Seq(topComp("c1b"), topComp("c2b")/*, topComp("bsum")*/), "B", "cB", Some("_OUT"), Some("_IN"))
    )
    val check =
      s"""circuit Top :
         |  module $top :
         |    output out: UInt<16>
         |    inst cA of A
         |    inst cB of B
         |    node asum = add(cA.c1a_out_OUT, cA.c2a_out_OUT)
         |    node bsum = add(cB.c1b_out_OUT, cB.c2b_out_OUT)
         |    out <= add(asum, bsum)
         |  module A :
         |    output c1a_out_OUT: UInt<8>
         |    output c2a_out_OUT: UInt<8>
         |    inst c1a of Const1A
         |    inst c2a of Const2A
         |    c1a_out_OUT <= c1a.out
         |    c2a_out_OUT <= c2a.out
         |  module B :
         |    output c1b_out_OUT: UInt<8>
         |    output c2b_out_OUT: UInt<8>
         |    inst c1b of Const1B
         |    inst c2b of Const2B
         |    c1b_out_OUT <= c1b.out
         |    c2b_out_OUT <= c2b.out
         |  module Const1A :
         |    output out: UInt<8>
         |    out <= UInt(1)
         |  module Const2A :
         |    output out: UInt<8>
         |    out <= UInt(2)
         |  module Const1B :
         |    output out: UInt<8>
         |    out <= UInt(1)
         |  module Const2B :
         |    output out: UInt<8>
         |    out <= UInt(2)
      """.stripMargin
    execute(input, check, groups)
  }
  "The two sets of instances" should "be grouped with their nodes" in {
    val input =
      s"""circuit $top :
         |  module $top :
         |    output out: UInt<16>
         |    inst c1a of Const1A
         |    inst c2a of Const2A
         |    inst c1b of Const1B
         |    inst c2b of Const2B
         |    node asum = add(c1a.out, c2a.out)
         |    node bsum = add(c1b.out, c2b.out)
         |    out <= add(asum, bsum)
         |  module Const1A :
         |    output out: UInt<8>
         |    out <= UInt(1)
         |  module Const2A :
         |    output out: UInt<8>
         |    out <= UInt(2)
         |  module Const1B :
         |    output out: UInt<8>
         |    out <= UInt(1)
         |  module Const2B :
         |    output out: UInt<8>
         |    out <= UInt(2)
      """.stripMargin
    val groups = Seq(
      GroupAnnotation(Seq(topComp("c1a"), topComp("c2a"), topComp("asum")), "A", "cA", Some("_OUT"), Some("_IN")),
      GroupAnnotation(Seq(topComp("c1b"), topComp("c2b"), topComp("bsum")), "B", "cB", Some("_OUT"), Some("_IN"))
    )
    val check =
      s"""circuit Top :
         |  module $top :
         |    output out: UInt<16>
         |    inst cA of A
         |    inst cB of B
         |    out <= add(cA.asum_OUT, cB.bsum_OUT)
         |  module A :
         |    output asum_OUT: UInt<9>
         |    inst c1a of Const1A
         |    inst c2a of Const2A
         |    node asum = add(c1a.out, c2a.out)
         |    asum_OUT <= asum
         |  module B :
         |    output bsum_OUT: UInt<9>
         |    inst c1b of Const1B
         |    inst c2b of Const2B
         |    node bsum = add(c1b.out, c2b.out)
         |    bsum_OUT <= bsum
         |  module Const1A :
         |    output out: UInt<8>
         |    out <= UInt(1)
         |  module Const2A :
         |    output out: UInt<8>
         |    out <= UInt(2)
         |  module Const1B :
         |    output out: UInt<8>
         |    out <= UInt(1)
         |  module Const2B :
         |    output out: UInt<8>
         |    out <= UInt(2)
      """.stripMargin
    execute(input, check, groups)
  }

  "The two sets of instances" should "be grouped with one not grouped" in {
    val input =
      s"""circuit $top :
         |  module $top :
         |    output out: UInt<16>
         |    inst c1a of Const1A
         |    inst c2a of Const2A
         |    inst c1b of Const1B
         |    inst c2b of Const2B
         |    node asum = add(c1a.out, c2a.out)
         |    node bsum = add(c1b.out, c2b.out)
         |    inst pass of PassThrough
         |    pass.in <= add(asum, bsum)
         |    out <= pass.out
         |  module Const1A :
         |    output out: UInt<8>
         |    out <= UInt(1)
         |  module Const2A :
         |    output out: UInt<8>
         |    out <= UInt(2)
         |  module Const1B :
         |    output out: UInt<8>
         |    out <= UInt(1)
         |  module Const2B :
         |    output out: UInt<8>
         |    out <= UInt(2)
         |  module PassThrough :
         |    input in: UInt
         |    output out: UInt
         |    out <= in
      """.stripMargin
    val groups = Seq(
      GroupAnnotation(Seq(topComp("c1a"), topComp("c2a"), topComp("asum")), "A", "cA", Some("_OUT"), Some("_IN")),
      GroupAnnotation(Seq(topComp("c1b"), topComp("c2b"), topComp("bsum")), "B", "cB", Some("_OUT"), Some("_IN"))
    )
    val check =
      s"""circuit Top :
         |  module $top :
         |    output out: UInt<16>
         |    inst cA of A
         |    inst cB of B
         |    inst pass of PassThrough
         |    out <= pass.out
         |    pass.in <= add(cA.asum_OUT, cB.bsum_OUT)
         |  module A :
         |    output asum_OUT: UInt<9>
         |    inst c1a of Const1A
         |    inst c2a of Const2A
         |    node asum = add(c1a.out, c2a.out)
         |    asum_OUT <= asum
         |  module B :
         |    output bsum_OUT: UInt<9>
         |    inst c1b of Const1B
         |    inst c2b of Const2B
         |    node bsum = add(c1b.out, c2b.out)
         |    bsum_OUT <= bsum
         |  module Const1A :
         |    output out: UInt<8>
         |    out <= UInt(1)
         |  module Const2A :
         |    output out: UInt<8>
         |    out <= UInt(2)
         |  module Const1B :
         |    output out: UInt<8>
         |    out <= UInt(1)
         |  module Const2B :
         |    output out: UInt<8>
         |    out <= UInt(2)
         |  module PassThrough :
         |    input in: UInt<10>
         |    output out: UInt<10>
         |    out <= in
      """.stripMargin
    execute(input, check, groups)
  }

  "The two sets of instances" should "be grouped with a connection between them" in {
    val input =
      s"""circuit $top :
         |  module $top :
         |    input in: UInt<16>
         |    output out: UInt<16>
         |    node first = in
         |    node second = not(first)
         |    out <= second
      """.stripMargin
    val groups = Seq(
      GroupAnnotation(Seq(topComp("first")), "First", "first"),
      GroupAnnotation(Seq(topComp("second")), "Second", "second")
    )
    val check =
      s"""circuit $top :
         |  module $top :
         |    input in: UInt<16>
         |    output out: UInt<16>
         |    inst first_0 of First
         |    inst second_0 of Second
         |    first_0.in <= in
         |    second_0.first <= first_0.first_0
         |    out <= second_0.second_0
         |  module First :
         |    input in: UInt<16>
         |    output first_0: UInt<16>
         |    node first = in
         |    first_0 <= first
         |  module Second :
         |    input first: UInt<16>
         |    output second_0: UInt<16>
         |    node second = not(first)
         |    second_0 <= second
      """.stripMargin
    execute(input, check, groups)
  }
}
