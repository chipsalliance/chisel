// See LICENSE for license details.

package firrtlTests
package transform

import java.io.StringWriter

import org.scalatest.FlatSpec
import org.scalatest.Matchers
import org.scalatest.junit.JUnitRunner

import firrtl.ir.Circuit
import firrtl.{Parser, AnnotationMap}
import firrtl.passes.PassExceptions
import firrtl.annotations.{
   Named,
   CircuitName,
   ModuleName,
   Annotation
}
import firrtl.transforms.{DedupModules, NoDedupAnnotation}


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
      val writer = new StringWriter()
      val aMap = new AnnotationMap(Nil)
      execute(writer, aMap, input, check)
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
      val writer = new StringWriter()
      val aMap = new AnnotationMap(Nil)
      execute(writer, aMap, input, check)
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
           |  module A : @[yy 2:2 xx 1:1]
           |    output x: UInt<1> @[yy 2:2 xx 1:1]
           |    inst b of B @[yy 2:2 xx 1:1]
           |    x <= b.x @[yy 2:2 xx 1:1]
           |  module B :
           |    output x: UInt<1>
           |    x <= UInt(1)
           """.stripMargin
      val writer = new StringWriter()
      val aMap = new AnnotationMap(Nil)
      execute(writer, aMap, input, check)
   }
   "The module B, but not A, with comments" should "be deduped if not annotated" in {
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
      val writer = new StringWriter()
      val aMap = new AnnotationMap(Seq(NoDedupAnnotation(ModuleName("A", CircuitName("Top")))))
      execute(writer, aMap, input, check)
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
