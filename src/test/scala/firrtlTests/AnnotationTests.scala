package firrtlTests

import java.io.{Writer, StringWriter}

import org.scalatest.FlatSpec
import org.scalatest.Matchers
import org.scalatest.junit.JUnitRunner

import firrtl.ir.Circuit
import firrtl.Parser
import firrtl.{
   ResolveAndCheck,
   RenameMap,
   Compiler,
   CompilerResult,
   VerilogCompiler
}
import firrtl.Annotations.{
   TransID,
   Named,
   CircuitName,
   ModuleName,
   ComponentName,
   AnnotationException,
   Annotation,
   Strict,
   Rigid,
   Firm,
   Loose,
   Sticky,
   Insistent,
   Fickle,
   Unstable,
   AnnotationMap
}

/**
 * An example methodology for testing Firrtl annotations.
 */
trait AnnotationSpec extends LowTransformSpec {
  // Dummy transform
  def transform = new ResolveAndCheck()

  // Check if Annotation Exception is thrown
  override def failingexecute(writer: Writer, annotations: AnnotationMap, input: String) = {
    intercept[AnnotationException] {
      compile(parse(input), annotations, writer)
    }
  }
  def execute(writer: Writer, annotations: AnnotationMap, input: String, check: Annotation) = {
    val cr = compile(parse(input), annotations, writer)
    (cr.annotationMap.annotations.head) should be (check)
  }
}


/**
 * Tests for Annotation Permissibility and Tenacity
 *
 * WARNING(izraelevitz): Not a complete suite of tests, requires the LowerTypes
 * pass and ConstProp pass to correctly populate its RenameMap before Strict, Rigid, Firm,
 * Unstable, Fickle, and Insistent can be tested.
 */
class AnnotationTests extends AnnotationSpec with Matchers {
  def getAMap (a: Annotation): AnnotationMap = new AnnotationMap(Seq(a))
  val tID = TransID(1)
  val input =
    """circuit Top :
       |  module Top :
       |    input a : UInt<1>[2]
       |    input b : UInt<1>
       |    node c = b""".stripMargin
  val mName = ModuleName("Top", CircuitName("Top"))
  val aName = ComponentName("a", mName)
  val bName = ComponentName("b", mName)
  val cName = ComponentName("c", mName)

  "Loose and Sticky annotation on a node" should "pass through" in {
    case class TestAnnotation(target: Named, tID: TransID) extends Annotation with Loose with Sticky {
      def duplicate(to: Named) = this.copy(target=to)
    }
    val w = new StringWriter()
    val ta = TestAnnotation(cName, tID)
    execute(w, getAMap(ta), input, ta)
  }
}
