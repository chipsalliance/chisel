package firrtlTests

import java.io.{Writer, StringWriter}

import org.scalatest.FlatSpec
import org.scalatest.Matchers
import org.scalatest.junit.JUnitRunner

import firrtl.ir.Circuit
import firrtl.Parser
import firrtl.{
   CircuitState,
   ResolveAndCheck,
   RenameMap,
   Compiler,
   ChirrtlForm,
   LowForm,
   VerilogCompiler,
   Transform
}
import firrtl.Annotations.{
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
  def transform = new CustomResolveAndCheck(LowForm)

  // Check if Annotation Exception is thrown
  override def failingexecute(writer: Writer, annotations: AnnotationMap, input: String) = {
    intercept[AnnotationException] {
      compile(CircuitState(parse(input), ChirrtlForm, Some(annotations)), writer)
    }
  }
  def execute(writer: Writer, annotations: AnnotationMap, input: String, check: Annotation) = {
    val cr = compile(CircuitState(parse(input), ChirrtlForm, Some(annotations)), writer)
    (cr.annotations.get.annotations.head) should be (check)
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
    case class TestAnnotation(target: Named) extends Annotation with Loose with Sticky {
      def duplicate(to: Named) = this.copy(target=to)
      def transform = classOf[Transform]
    }
    val w = new StringWriter()
    val ta = TestAnnotation(cName)
    execute(w, getAMap(ta), input, ta)
  }
}
