// See LICENSE for license details.

package firrtlTests

import java.io._
import org.scalatest._
import org.scalatest.prop._
import firrtl.Parser
import firrtl.ir.Circuit
import firrtl.passes.{Pass,ToWorkingIR,CheckHighForm,ResolveKinds,InferTypes,CheckTypes,PassExceptions,InferWidths,CheckWidths,ResolveGenders,CheckGenders}

class CheckSpec extends FlatSpec with Matchers {
  "Connecting bundles of different types" should "throw an exception" in {
    val passes = Seq(
      ToWorkingIR,
      CheckHighForm)
    val input =
      """circuit Unit :
        |  module Unit :
        |    mem m :
        |      data-type => {a : {b : {flip c : UInt<32>}}}
        |      depth => 32
        |      read-latency => 0
        |      write-latency => 1""".stripMargin
    intercept[CheckHighForm.MemWithFlipException] {
      passes.foldLeft(Parser.parse(input.split("\n").toIterator)) {
        (c: Circuit, p: Pass) => p.run(c)
      }
    }
  }
  "Instance loops a -> b -> a" should "be detected" in {
    val passes = Seq(
      ToWorkingIR,
      CheckHighForm)
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
      passes.foldLeft(Parser.parse(input.split("\n").toIterator)) {
        (c: Circuit, p: Pass) => p.run(c)
      }
    }
  }

  "Instance loops a -> b -> c -> a" should "be detected" in {
    val passes = Seq(
      ToWorkingIR,
      CheckHighForm)
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
      passes.foldLeft(Parser.parse(input.split("\n").toIterator)) {
        (c: Circuit, p: Pass) => p.run(c)
      }
    }
  }

  "Instance loops a -> a" should "be detected" in {
    val passes = Seq(
      ToWorkingIR,
      CheckHighForm)
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
      passes.foldLeft(Parser.parse(input.split("\n").toIterator)) {
        (c: Circuit, p: Pass) => p.run(c)
      }
    }
  }

  "Instance loops should not have false positives" should "be detected" in {
    val passes = Seq(
      ToWorkingIR,
      CheckHighForm)
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
    passes.foldLeft(Parser.parse(input.split("\n").toIterator)) {
      (c: Circuit, p: Pass) => p.run(c)
    }

  }

  "Clock Types" should "be connectable" in {
    val passes = Seq(
      ToWorkingIR,
      CheckHighForm,
      ResolveKinds,
      InferTypes,
      CheckTypes,
      ResolveGenders,
      CheckGenders,
      InferWidths,
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
          |    input clk : Clock
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
    passes.foldLeft(Parser.parse(input.split("\n").toIterator)) {
      (c: Circuit, p: Pass) => p.run(c)
    }
  }

}
