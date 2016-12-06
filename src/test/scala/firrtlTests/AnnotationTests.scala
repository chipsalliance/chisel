// See LICENSE for license details.

package firrtlTests

import java.io.{File, FileWriter, StringWriter, Writer}

import firrtl.annotations.AnnotationYamlProtocol._
import firrtl.annotations._
import firrtl._
import firrtl.passes.InlineAnnotation
import firrtl.passes.memlib.PinAnnotation
import net.jcazevedo.moultingyaml._
import org.scalatest.Matchers

/**
 * An example methodology for testing Firrtl annotations.
 */
trait AnnotationSpec extends LowTransformSpec {
  // Dummy transform
  def transform = new CustomResolveAndCheck(LowForm)

  // Check if Annotation Exception is thrown
  override def failingexecute(writer: Writer, annotations: AnnotationMap, input: String): Exception = {
    intercept[AnnotationException] {
      compile(CircuitState(parse(input), ChirrtlForm, Some(annotations)), writer)
    }
  }
  def execute(writer: Writer, annotations: AnnotationMap, input: String, check: Annotation): Unit = {
    val cr = compile(CircuitState(parse(input), ChirrtlForm, Some(annotations)), writer)
    cr.annotations.get.annotations should be (Seq(check))
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
  def getAMap (a: Annotation): AnnotationMap = AnnotationMap(Seq(a))
  val input: String =
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
    val w = new StringWriter()
    val ta = Annotation(cName, classOf[Transform], "")
    execute(w, getAMap(ta), input, ta)
  }

  "Annotations" should "be readable from file" in {
    val annotationFile = new File("src/test/resources/annotations/SampleAnnotations.anno")
    val annotationsYaml = io.Source.fromFile(annotationFile).getLines().mkString("\n").parseYaml
    val annotationArray = annotationsYaml.convertTo[Array[Annotation]]
    annotationArray.length should be (9)
    annotationArray(0).targetString should be ("ModC")
    annotationArray(7).transformClass should be ("firrtl.passes.InlineInstances")
    val expectedValue = "TopOfDiamond\nWith\nSome new lines"
    annotationArray(7).value should be (expectedValue)
  }

  "Badly formatted serializations" should "return reasonable error messages" in {
    var badYaml =
      """
        |- transformClass: firrtl.passes.InlineInstances
        |  targetString: circuit.module..
        |  value: ModC.this params 16 32
      """.stripMargin.parseYaml

    var thrown = intercept[Exception] {
      badYaml.convertTo[Array[Annotation]]
    }
    thrown.getMessage should include ("Illegal component name")

    badYaml =
      """
        |- transformClass: firrtl.passes.InlineInstances
        |  targetString: .circuit.module.component
        |  value: ModC.this params 16 32
      """.stripMargin.parseYaml

    thrown = intercept[Exception] {
      badYaml.convertTo[Array[Annotation]]
    }
    thrown.getMessage should include ("Illegal circuit name")
  }

  "Round tripping annotations through text file" should "preserve annotations" in {
    val annos: Array[Annotation] = Seq(
      InlineAnnotation(CircuitName("fox")),
      InlineAnnotation(ModuleName("dog", CircuitName("bear"))),
      InlineAnnotation(ComponentName("chocolate", ModuleName("like", CircuitName("i")))),
      PinAnnotation(CircuitName("Pinniped"), Seq("sea-lion", "monk-seal"))
    ).toArray

    val annoFile = new File("temp-anno")
    val writer = new FileWriter(annoFile)
    writer.write(annos.toYaml.prettyPrint)
    writer.close()

    val yaml = io.Source.fromFile(annoFile).getLines().mkString("\n").parseYaml
    annoFile.delete()

    val readAnnos = yaml.convertTo[Array[Annotation]]

    annos.zip(readAnnos).foreach { case (beforeAnno, afterAnno) =>
      beforeAnno.targetString should be (afterAnno.targetString)
      beforeAnno.target should be (afterAnno.target)
      beforeAnno.transformClass should be (afterAnno.transformClass)
      beforeAnno.transform should be (afterAnno.transform)

      beforeAnno should be (afterAnno)
    }
  }
}
