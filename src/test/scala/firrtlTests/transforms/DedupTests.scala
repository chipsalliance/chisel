// See LICENSE for license details.

package firrtlTests
package transform

import firrtl.annotations._
import firrtl.transforms.{DedupModules}


/**
 * Tests inline instances transformation
 */
class DedupModuleTests extends HighTransformSpec {
  def transform = new DedupModules
  "The module A" should "be deduped" in {
     val input =
        """circuit Top :
          |  module Top :
          |    inst a1 of A
          |    inst a2 of A_
          |  module A :
          |    output x: UInt<1>
          |    x <= UInt(1)
          |  module A_ :
          |    output x: UInt<1>
          |    x <= UInt(1)
          """.stripMargin
     val check =
        """circuit Top :
          |  module Top :
          |    inst a1 of A
          |    inst a2 of A
          |  module A :
          |    output x: UInt<1>
          |    x <= UInt(1)
          """.stripMargin
     execute(input, check, Seq.empty)
  }
  "The module A and B" should "be deduped" in {
     val input =
        """circuit Top :
          |  module Top :
          |    inst a1 of A
          |    inst a2 of A_
          |  module A :
          |    output x: UInt<1>
          |    inst b of B
          |    x <= b.x
          |  module A_ :
          |    output x: UInt<1>
          |    inst b of B_
          |    x <= b.x
          |  module B :
          |    output x: UInt<1>
          |    x <= UInt(1)
          |  module B_ :
          |    output x: UInt<1>
          |    x <= UInt(1)
          """.stripMargin
     val check =
        """circuit Top :
          |  module Top :
          |    inst a1 of A
          |    inst a2 of A
          |  module A :
          |    output x: UInt<1>
          |    inst b of B
          |    x <= b.x
          |  module B :
          |    output x: UInt<1>
          |    x <= UInt(1)
          """.stripMargin
     execute(input, check, Seq.empty)
  }
  "The module A and B with comments" should "be deduped" in {
     val input =
        """circuit Top :
          |  module Top :
          |    inst a1 of A
          |    inst a2 of A_
          |  module A : @[yy 2:2]
          |    output x: UInt<1> @[yy 2:2]
          |    inst b of B @[yy 2:2]
          |    x <= b.x @[yy 2:2]
          |  module A_ : @[xx 1:1]
          |    output x: UInt<1> @[xx 1:1]
          |    inst b of B_ @[xx 1:1]
          |    x <= b.x @[xx 1:1]
          |  module B :
          |    output x: UInt<1>
          |    x <= UInt(1)
          |  module B_ :
          |    output x: UInt<1>
          |    x <= UInt(1)
          """.stripMargin
     val check =
        """circuit Top :
          |  module Top :
          |    inst a1 of A
          |    inst a2 of A
          |  module A : @[yy 2:2]
          |    output x: UInt<1> @[yy 2:2]
          |    inst b of B @[yy 2:2]
          |    x <= b.x @[yy 2:2]
          |  module B :
          |    output x: UInt<1>
          |    x <= UInt(1)
          """.stripMargin
     execute(input, check, Seq.empty)
  }
  "A_ but not A" should "be deduped if not annotated" in {
     val input =
        """circuit Top :
          |  module Top :
          |    inst a1 of A
          |    inst a2 of A_
          |  module A : @[yy 2:2]
          |    output x: UInt<1> @[yy 2:2]
          |    x <= UInt(1)
          |  module A_ : @[xx 1:1]
          |    output x: UInt<1> @[xx 1:1]
          |    x <= UInt(1)
          """.stripMargin
     val check =
        """circuit Top :
          |  module Top :
          |    inst a1 of A
          |    inst a2 of A_
          |  module A : @[yy 2:2]
          |    output x: UInt<1> @[yy 2:2]
          |    x <= UInt(1)
          |  module A_ : @[xx 1:1]
          |    output x: UInt<1> @[xx 1:1]
          |    x <= UInt(1)
          """.stripMargin
     execute(input, check, Seq(dontDedup("A")))
  }
  "The module A and A_" should "be deduped even with different port names and info, and annotations should remap" in {
     val input =
        """circuit Top :
          |  module Top :
          |    output out: UInt<1>
          |    inst a1 of A
          |    inst a2 of A_
          |    out <= and(a1.x, a2.y)
          |  module A : @[yy 2:2]
          |    output x: UInt<1> @[yy 2:2]
          |    x <= UInt(1)
          |  module A_ : @[xx 1:1]
          |    output y: UInt<1> @[xx 1:1]
          |    y <= UInt(1)
        """.stripMargin
     val check =
        """circuit Top :
          |  module Top :
          |    output out: UInt<1>
          |    inst a1 of A
          |    inst a2 of A
          |    out <= and(a1.x, a2.x)
          |  module A : @[yy 2:2]
          |    output x: UInt<1> @[yy 2:2]
          |    x <= UInt(1)
        """.stripMargin
     case class DummyAnnotation(target: ComponentName) extends SingleTargetAnnotation[ComponentName] {
       override def duplicate(n: ComponentName): Annotation = DummyAnnotation(n)
     }

     val mname = ModuleName("Top", CircuitName("Top"))
     val finalState = execute(input, check, Seq(DummyAnnotation(ComponentName("a2.y", mname))))

     finalState.annotations.collect({ case d: DummyAnnotation => d }).head should be(DummyAnnotation(ComponentName("a2.x", mname)))

  }
  "The module A and B" should "be deduped with the first module in order" in {
    val input =
      """circuit Top :
        |  module Top :
        |    inst a1 of A
        |    inst a2 of A_
        |  module A :
        |    output x: UInt<1>
        |    inst b of B_
        |    x <= b.x
        |  module A_ :
        |    output x: UInt<1>
        |    inst b of B
        |    x <= b.x
        |  module B :
        |    output x: UInt<1>
        |    x <= UInt(1)
        |  module B_ :
        |    output x: UInt<1>
        |    x <= UInt(1)
      """.stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    inst a1 of A
        |    inst a2 of A
        |  module A :
        |    output x: UInt<1>
        |    inst b of B
        |    x <= b.x
        |  module B :
        |    output x: UInt<1>
        |    x <= UInt(1)
      """.stripMargin
    execute(input, check, Seq.empty)
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
