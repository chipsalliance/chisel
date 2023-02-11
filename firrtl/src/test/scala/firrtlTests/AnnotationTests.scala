// SPDX-License-Identifier: Apache-2.0

package firrtlTests

import java.io.{File, FileWriter}
import firrtl.annotations._
import firrtl._
import firrtl.FileUtils
import firrtl.options.{Dependency, InputAnnotationFileAnnotation}
import firrtl.transforms.OptimizableExtModuleAnnotation
import firrtl.passes.InlineAnnotation
import firrtl.stage.{FirrtlSourceAnnotation, FirrtlStage}
import firrtl.util.BackendCompilationUtilities
import firrtl.testutils._
import org.scalatest.matchers.should.Matchers

object AnnotationTests {

  class DeletingTransform extends Transform {
    val inputForm = LowForm
    val outputForm = LowForm
    def execute(state: CircuitState) = state.copy(annotations = Seq())
  }

}

// Abstract but with lots of tests defined so that we can use the same tests
// for Legacy and newer Annotations
abstract class AnnotationTests extends LowFirrtlTransformSpec with Matchers with MakeCompiler {
  import AnnotationTests._

  def anno(s:    String, value: String = "this is a value", mod: String = "Top"): Annotation
  def manno(mod: String): Annotation

  "Annotation on a node" should "pass through" in {
    val input: String =
      """circuit Top :
        |  module Top :
        |    input a : UInt<1>[2]
        |    input b : UInt<1>
        |    node c = b""".stripMargin
    val ta = anno("c", "")
    val r = compile(input, Seq(ta))
    r.annotations.toSeq should contain(ta)
  }

  "Renaming" should "track deduplication" in {
    val input =
      """circuit Top :
        |  module Child :
        |    input x : UInt<32>
        |    output y : UInt<32>
        |    y <= x
        |  module Child_1 :
        |    input x : UInt<32>
        |    output y : UInt<32>
        |    y <= x
        |  module Top :
        |    input in : UInt<32>[2]
        |    output out : UInt<32>
        |    inst a of Child
        |    inst b of Child_1
        |    a.x <= in[0]
        |    b.x <= in[1]
        |    out <= tail(add(a.y, b.y), 1)
        |""".stripMargin
    val annos = Seq(
      anno("x", mod = "Child"),
      anno("y", mod = "Child_1"),
      manno("Child"),
      manno("Child_1")
    )
    val result = compile(input, annos)
    val resultAnno = result.annotations.toSeq
    resultAnno should contain(anno("x", mod = "Child"))
    resultAnno should contain(anno("y", mod = "Child"))
    resultAnno should contain(manno("Child"))
    resultAnno should not contain (anno("y", mod = "Child_1"))
    resultAnno should not contain (manno("Child_1"))
  }

  "AnnotationUtils.toNamed" should "invert Named.serialize" in {
    val x = ComponentName("component", ModuleName("module", CircuitName("circuit")))
    val y = AnnotationUtils.toNamed(x.serialize)
    require(x == y)
  }

}

class JsonAnnotationTests extends AnnotationTests {
  // Helper annotations
  case class SimpleAnno(target: ComponentName, value: String) extends SingleTargetAnnotation[ComponentName] {
    def duplicate(n: ComponentName) = this.copy(target = n)
  }
  case class ModuleAnno(target: ModuleName) extends SingleTargetAnnotation[ModuleName] {
    def duplicate(n: ModuleName) = this.copy(target = n)
  }

  def anno(s: String, value: String = "this is a value", mod: String = "Top"): SimpleAnno =
    SimpleAnno(ComponentName(s, ModuleName(mod, CircuitName("Top"))), value)
  def manno(mod: String): Annotation = ModuleAnno(ModuleName(mod, CircuitName("Top")))

  "Round tripping annotations through text file" should "preserve annotations" in {
    val annos: Array[Annotation] = Seq(
      InlineAnnotation(CircuitName("fox")),
      InlineAnnotation(ModuleName("dog", CircuitName("bear"))),
      InlineAnnotation(ComponentName("chocolate", ModuleName("like", CircuitName("i")))),
      InlineAnnotation(ComponentName("chocolate.frog", ModuleName("like", CircuitName("i"))))
    ).toArray

    val annoFile = new File("temp-anno")
    val writer = new FileWriter(annoFile)
    writer.write(JsonProtocol.serialize(annos))
    writer.close()

    val text = FileUtils.getText(annoFile)
    annoFile.delete()

    val readAnnos = JsonProtocol.deserializeTry(text).get

    annos should be(readAnnos)
  }

  private def setupManager(annoFileText: Option[String]): Driver.Arg = {
    val source = """
                   |circuit test :
                   |  module test :
                   |    input x : UInt<1>
                   |    output z : UInt<1>
                   |    z <= x
                   |    node y = x""".stripMargin
    val testDir = BackendCompilationUtilities.createTestDirectory(this.getClass.getSimpleName)
    val annoFile = new File(testDir, "anno.json")

    annoFileText.foreach { text =>
      val w = new FileWriter(annoFile)
      w.write(text)
      w.close()
    }

    (
      Array("--target-dir", testDir.getPath),
      Seq(FirrtlSourceAnnotation(source), InputAnnotationFileAnnotation(annoFile.getPath))
    )
  }

  private object Driver {
    type Arg = (Array[String], AnnotationSeq)
    def execute(args: Arg) = ((new FirrtlStage).execute _).tupled(args)
  }

  "Annotation file not found" should "give a reasonable error message" in {
    val manager = setupManager(None)

    an[AnnotationFileNotFoundException] shouldBe thrownBy {
      Driver.execute(manager)
    }
  }

  "Annotation class not found" should "give a reasonable error message" in {
    val anno = """
                 |[
                 |  {
                 |    "class":"ThisClassDoesNotExist",
                 |    "target":"test.test.y"
                 |  }
                 |] """.stripMargin
    val manager = setupManager(Some(anno))

    the[Exception] thrownBy Driver.execute(manager) should matchPattern {
      case InvalidAnnotationFileException(_, _: UnrecogizedAnnotationsException) =>
    }
  }

  "Malformed annotation file" should "give a reasonable error message" in {
    val anno = """
                 |[
                 |  {
                 |    "class":
                 |    "target":"test.test.y"
                 |  }
                 |] """.stripMargin
    val manager = setupManager(Some(anno))

    the[Exception] thrownBy Driver.execute(manager) should matchPattern {
      case InvalidAnnotationFileException(_, _: InvalidAnnotationJSONException) =>
    }
  }

  "Non-array annotation file" should "give a reasonable error message" in {
    val anno = """
                 |{
                 |  "class":"firrtl.transforms.DontTouchAnnotation",
                 |  "target":"test.test.y"
                 |}
                 |""".stripMargin
    val manager = setupManager(Some(anno))

    the[Exception] thrownBy Driver.execute(manager) should matchPattern {
      case InvalidAnnotationFileException(_, InvalidAnnotationJSONException(msg)) if msg.contains("JObject") =>
    }
  }

  object DoNothingTransform extends Transform {
    override def inputForm:  CircuitForm = UnknownForm
    override def outputForm: CircuitForm = UnknownForm

    def execute(state: CircuitState): CircuitState = state
  }

  "annotation order" should "should be preserved" in {
    val annos = Seq(anno("a"), anno("b"), anno("c"), anno("d"), anno("e"))
    val input: String =
      """circuit Top :
        |  module Top :
        |    input a : UInt<1>
        |    node b = c""".stripMargin
    val cr = DoNothingTransform.runTransform(CircuitState(parse(input), ChirrtlForm, annos))
    cr.annotations.toSeq shouldEqual annos
  }

  "fully qualified class name that is undeserializable" should "give an invalid json exception" in {
    val anno = """
                 |[
                 |  {
                 |    "class":"firrtlTests.MyUnserAnno",
                 |    "box":"7"
                 |  }
                 |] """.stripMargin

    val manager = setupManager(Some(anno))
    the[Exception] thrownBy Driver.execute(manager) should matchPattern {
      case InvalidAnnotationFileException(_, _: InvalidAnnotationJSONException) =>
    }
  }

  "unqualified class name" should "give an unrecognized annotation exception" in {
    val anno = """
                 |[
                 |  {
                 |    "class":"MyUnserAnno"
                 |    "box":"7"
                 |  }
                 |] """.stripMargin
    val manager = setupManager(Some(anno))
    the[Exception] thrownBy Driver.execute(manager) should matchPattern {
      case InvalidAnnotationFileException(_, _: UnrecogizedAnnotationsException) =>
    }
  }
}

/* These are used by the last two tests. It is outside the main test to keep the qualified name simpler*/
class UnserBox(val x: Int)
case class MyUnserAnno(box: UnserBox) extends NoTargetAnnotation
