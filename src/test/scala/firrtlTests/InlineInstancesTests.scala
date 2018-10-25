// See LICENSE for license details.

package firrtlTests

import org.scalatest.FlatSpec
import org.scalatest.Matchers
import org.scalatest.junit.JUnitRunner
import firrtl.ir.Circuit
import firrtl.Parser
import firrtl.passes.PassExceptions
import firrtl.annotations.{Annotation, CircuitName, ComponentName, ModuleName, Named}
import firrtl.passes.{InlineAnnotation, InlineInstances}
import logger.{LogLevel, Logger}
import logger.LogLevel.Debug


/**
 * Tests inline instances transformation
 */
class InlineInstancesTests extends LowTransformSpec {
  def transform = new InlineInstances
	def inline(mod: String): Annotation = {
	  val parts = mod.split('.')
		val modName = ModuleName(parts.head, CircuitName("Top")) // If this fails, bad input
		val name = if (parts.size == 1) modName
							 else ComponentName(parts.tail.mkString("."), modName)
    InlineAnnotation(name)
  }
   // Set this to debug, this will apply to all tests
   // Logger.setLevel(this.getClass, Debug)
   "The module Inline" should "be inlined" in {
      val input =
         """circuit Top :
           |  module Top :
           |    input a : UInt<32>
           |    output b : UInt<32>
           |    inst i of Inline
           |    i.a <= a
           |    b <= i.b
           |  module Inline :
           |    input a : UInt<32>
           |    output b : UInt<32>
           |    b <= a""".stripMargin
      val check =
         """circuit Top :
           |  module Top :
           |    input a : UInt<32>
           |    output b : UInt<32>
           |    wire i_a : UInt<32>
           |    wire i_b : UInt<32>
           |    i_b <= i_a
           |    b <= i_b
           |    i_a <= a""".stripMargin
      execute(input, check, Seq(inline("Inline")))
   }

   "The all instances of Simple" should "be inlined" in {
      val input =
         """circuit Top :
           |  module Top :
           |    input a : UInt<32>
           |    output b : UInt<32>
           |    inst i0 of Simple
           |    inst i1 of Simple
           |    i0.a <= a
           |    i1.a <= i0.b
           |    b <= i1.b
           |  module Simple :
           |    input a : UInt<32>
           |    output b : UInt<32>
           |    b <= a""".stripMargin
      val check =
         """circuit Top :
           |  module Top :
           |    input a : UInt<32>
           |    output b : UInt<32>
           |    wire i0_a : UInt<32>
           |    wire i0_b : UInt<32>
           |    i0_b <= i0_a
           |    wire i1_a : UInt<32>
           |    wire i1_b : UInt<32>
           |    i1_b <= i1_a
           |    b <= i1_b
           |    i0_a <= a
           |    i1_a <= i0_b""".stripMargin
      execute(input, check, Seq(inline("Simple")))
   }

   "Only one instance of Simple" should "be inlined" in {
      val input =
         """circuit Top :
           |  module Top :
           |    input a : UInt<32>
           |    output b : UInt<32>
           |    inst i0 of Simple
           |    inst i1 of Simple
           |    i0.a <= a
           |    i1.a <= i0.b
           |    b <= i1.b
           |  module Simple :
           |    input a : UInt<32>
           |    output b : UInt<32>
           |    b <= a""".stripMargin
      val check =
         """circuit Top :
           |  module Top :
           |    input a : UInt<32>
           |    output b : UInt<32>
           |    wire i0_a : UInt<32>
           |    wire i0_b : UInt<32>
           |    i0_b <= i0_a
           |    inst i1 of Simple
           |    b <= i1.b
           |    i0_a <= a
           |    i1.a <= i0_b
           |  module Simple :
           |    input a : UInt<32>
           |    output b : UInt<32>
           |    b <= a""".stripMargin
      execute(input, check, Seq(inline("Top.i0")))
   }

   "All instances of A" should "be inlined" in {
      val input =
         """circuit Top :
           |  module Top :
           |    input a : UInt<32>
           |    output b : UInt<32>
           |    inst i0 of A
           |    inst i1 of B
           |    i0.a <= a
           |    i1.a <= i0.b
           |    b <= i1.b
           |  module A :
           |    input a : UInt<32>
           |    output b : UInt<32>
           |    b <= a
           |  module B :
           |    input a : UInt<32>
           |    output b : UInt<32>
           |    inst i of A
           |    i.a <= a
           |    b <= i.b""".stripMargin
      val check =
         """circuit Top :
           |  module Top :
           |    input a : UInt<32>
           |    output b : UInt<32>
           |    wire i0_a : UInt<32>
           |    wire i0_b : UInt<32>
           |    i0_b <= i0_a
           |    inst i1 of B
           |    b <= i1.b
           |    i0_a <= a
           |    i1.a <= i0_b
           |  module B :
           |    input a : UInt<32>
           |    output b : UInt<32>
           |    wire i_a : UInt<32>
           |    wire i_b : UInt<32>
           |    i_b <= i_a
           |    b <= i_b
           |    i_a <= a""".stripMargin
      execute(input, check, Seq(inline("A")))
   }

   "Non-inlined instances" should "still prepend prefix" in {
      val input =
         """circuit Top :
           |  module Top :
           |    input a : UInt<32>
           |    output b : UInt<32>
           |    inst i of A
           |    i.a <= a
           |    b <= i.b
           |  module A :
           |    input a : UInt<32>
           |    output b : UInt<32>
           |    inst i of B
           |    i.a <= a
           |    b <= i.b
           |  module B :
           |    input a : UInt<32>
           |    output b : UInt<32>
           |    b <= a""".stripMargin
      val check =
         """circuit Top :
           |  module Top :
           |    input a : UInt<32>
           |    output b : UInt<32>
           |    wire i_a : UInt<32>
           |    wire i_b : UInt<32>
           |    inst i_i of B
           |    i_b <= i_i.b
           |    i_i.a <= i_a
           |    b <= i_b
           |    i_a <= a
           |  module B :
           |    input a : UInt<32>
           |    output b : UInt<32>
           |    b <= a""".stripMargin
      execute(input, check, Seq(inline("A")))
   }

  "A module with nested inlines" should "still prepend prefixes" in {
    val input =
      """|circuit Top:
         |  module Top:
         |    inst foo of Foo
         |  module Foo:
         |    inst bar of Bar
         |    inst baz of Bar
         |    node foo = UInt<1>("h0")
         |  module Bar:
         |    node bar = UInt<1>("h0")
         |""".stripMargin
    val check =
      """|circuit Top:
         |  module Top:
         |    node foo_bar_bar = UInt<1>("h0")
         |    inst foo_baz of Bar
         |    node foo_foo = UInt<1>("h0")
         |  module Bar:
         |    node bar = UInt<1>("h0")
         |""".stripMargin
    execute(input, check, Seq(inline("Foo"), inline("Foo.bar")))
  }

  "An inlined module" should "NOT be prefix unique" in {
    val input =
      """|circuit Top:
         |  module Top:
         |    inst a of A
         |    node a_foo = UInt<1>("h0")
         |    node a__bar = UInt<1>("h0")
         |  module A:
         |    node bar = UInt<1>("h0")
         |""".stripMargin
    val check =
      """|circuit Top:
         |  module Top:
         |    node a_bar = UInt<1>("h0")
         |    node a_foo = UInt<1>("h0")
         |    node a__bar = UInt<1>("h0")
         |""".stripMargin
    execute(input, check, Seq(inline("A")))
  }

  /* This test is mutually exclusive with the above */
  ignore should "be prefix unique" in {
    val input =
      """|circuit Top:
         |  module Top:
         |    inst a of A
         |    node a_foo = UInt<1>("h0")
         |    node a__bar = UInt<1>("h0")
         |  module A:
         |    node bar = UInt<1>("h0")
         |""".stripMargin
    val check =
      """|circuit Top:
         |  module Top:
         |    node a___bar = UInt<1>("h0")
         |    node a_foo = UInt<1>("h0")
         |    node a__bar = UInt<1>("h0")
         |""".stripMargin
    execute(input, check, Seq(inline("A")))
  }

  it should "uniquify sanely" in {
    val input =
      """|circuit Top:
         |  module Top:
         |    inst foo of Foo
         |    node foo_ = UInt<1>("h0")
         |    node foo__bar = UInt<1>("h0")
         |  module Foo:
         |    inst bar of Bar
         |    inst baz of Bar
         |    node foo = UInt<1>("h0")
         |  module Bar:
         |    node bar = UInt<1>("h0")
         |""".stripMargin
    val check =
      """|circuit Top:
         |  module Top:
         |    node foo__bar_bar = UInt<1>("h0")
         |    inst foo__baz of Bar
         |    node foo__foo = UInt<1>("h0")
         |    node foo_ = UInt<1>("h0")
         |    node foo__bar = UInt<1>("h0")
         |  module Bar:
         |    node bar = UInt<1>("h0")
         |""".stripMargin
    execute(input, check, Seq(inline("Foo"), inline("Foo.bar")))
  }

   // ---- Errors ----
   // 1) ext module
   "External module" should "not be inlined" in {
      val input =
         """circuit Top :
           |  module Top :
           |    input a : UInt<32>
           |    output b : UInt<32>
           |    inst i of A
           |    i.a <= a
           |    b <= i.b
           |  extmodule A :
           |    input a : UInt<32>
           |    output b : UInt<32>""".stripMargin
      failingexecute(input, Seq(inline("A")))
   }
   // 2) ext instance
   "External instance" should "not be inlined" in {
      val input =
         """circuit Top :
           |  module Top :
           |    input a : UInt<32>
           |    output b : UInt<32>
           |    inst i of A
           |    i.a <= a
           |    b <= i.b
           |  extmodule A :
           |    input a : UInt<32>
           |    output b : UInt<32>""".stripMargin
      failingexecute(input, Seq(inline("A")))
   }
   // 3) no module
   "Inlined module" should "exist" in {
      val input =
         """circuit Top :
           |  module Top :
           |    input a : UInt<32>
           |    output b : UInt<32>
           |    b <= a""".stripMargin
      failingexecute(input, Seq(inline("A")))
   }
   // 4) no inst
   "Inlined instance" should "exist" in {
      val input =
         """circuit Top :
           |  module Top :
           |    input a : UInt<32>
           |    output b : UInt<32>
           |    b <= a""".stripMargin
      failingexecute(input, Seq(inline("A")))
   }

  "Jack's Bug" should "not fail" in {

    val input = """circuit Top :
                  |  module Top :
                  |    input a : UInt<32>
                  |    output b : UInt<32>
                  |    inst i of Inline
                  |    i.a <= a
                  |    b <= i.b
                  |  module Inline :
                  |    input a : UInt<32>
                  |    output b : UInt<32>
                  |    inst child of InlineChild
                  |    child.a <= a
                  |    b <= child.b
                  |  module InlineChild :
                  |    input a : UInt<32>
                  |    output b : UInt<32>
                  |    b <= a""".stripMargin
    val check = """circuit Top :
                  |  module Top :
                  |    input a : UInt<32>
                  |    output b : UInt<32>
                  |    wire i_a : UInt<32>
                  |    wire i_b : UInt<32>
                  |    inst i_child of InlineChild
                  |    i_b <= i_child.b
                  |    i_child.a <= i_a
                  |    b <= i_b
                  |    i_a <= a
                  |  module InlineChild :
                  |    input a : UInt<32>
                  |    output b : UInt<32>
                  |    b <= a""".stripMargin
    execute(input, check, Seq(inline("Inline")))
  }
}

// Execution driven tests for inlining modules
// TODO(izraelevitz) fix this test
//class InlineInstancesIntegrationSpec extends FirrtlPropSpec {
//  // Shorthand for creating annotations to inline modules
//  def inlineModules(names: Seq[String]): Seq[CircuitAnnotation] =
//    Seq(StickyCircuitAnnotation(InlineCAKind, names.map(n => ModuleName(n) -> TagAnnotation).toMap))
//
//  case class Test(name: String, dir: String, ann: Seq[CircuitAnnotation])
//
//  val runTests = Seq(
//    Test("GCDTester", "/integration", inlineModules(Seq("DecoupledGCD")))
//  )
//
//  runTests foreach { test =>
//    property(s"${test.name} should execute correctly with inlining") {
//      println(s"Got annotations ${test.ann}")
//      runFirrtlTest(test.name, test.dir, test.ann)
//    }
//  }
//}
