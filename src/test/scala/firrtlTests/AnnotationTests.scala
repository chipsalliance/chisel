// See LICENSE for license details.

package firrtlTests

import java.io.{File, FileWriter}

import firrtl.annotations.AnnotationYamlProtocol._
import firrtl.annotations._
import firrtl._
import firrtl.FileUtils
import firrtl.transforms.OptimizableExtModuleAnnotation
import firrtl.passes.InlineAnnotation
import firrtl.passes.memlib.PinAnnotation
import firrtl.util.BackendCompilationUtilities
import net.jcazevedo.moultingyaml._
import org.scalatest.Matchers

/**
 * An example methodology for testing Firrtl annotations.
 */
trait AnnotationSpec extends LowTransformSpec {
  // Dummy transform
  def transform = new ResolveAndCheck

  // Check if Annotation Exception is thrown
  override def failingexecute(input: String, annotations: Seq[Annotation]): Exception = {
    intercept[AnnotationException] {
      compile(CircuitState(parse(input), ChirrtlForm, annotations), Seq.empty)
    }
  }
  def execute(input: String, check: Annotation, annotations: Seq[Annotation]): Unit = {
    val cr = compile(CircuitState(parse(input), ChirrtlForm, annotations), Seq.empty)
    cr.annotations.toSeq should contain (check)
  }
}

// Abstract but with lots of tests defined so that we can use the same tests
// for Legacy and newer Annotations
abstract class AnnotationTests extends AnnotationSpec with Matchers {
  def anno(s: String, value: String ="this is a value", mod: String = "Top"): Annotation
  def manno(mod: String): Annotation

  "Annotation on a node" should "pass through" in {
    val input: String =
      """circuit Top :
         |  module Top :
         |    input a : UInt<1>[2]
         |    input b : UInt<1>
         |    node c = b""".stripMargin
    val ta = anno("c", "")
    execute(input, ta, Seq(ta))
  }

  "Deleting annotations" should "create a DeletedAnnotation" in {
    val compiler = new VerilogCompiler
    val input =
     """circuit Top :
        |  module Top :
        |    input in: UInt<3>
        |""".stripMargin
    class DeletingTransform extends Transform {
      val inputForm = LowForm
      val outputForm = LowForm
      def execute(state: CircuitState) = state.copy(annotations = Seq())
    }
    val transform = new DeletingTransform
    val tname = transform.name
    val inlineAnn = InlineAnnotation(CircuitName("Top"))
    val result = compiler.compile(CircuitState(parse(input), ChirrtlForm, Seq(inlineAnn)), Seq(transform))
    result.annotations.head should matchPattern {
      case DeletedAnnotation(`tname`, `inlineAnn`) =>
    }
    val exception = (intercept[Exception] {
      result.getEmittedCircuit
    })
    val deleted = result.deletedAnnotations
    exception.getMessage should be (s"No EmittedCircuit found! Did you delete any annotations?\n$deleted")
  }

  "Renaming" should "propagate in Lowering of memories" in {
    val compiler = new VerilogCompiler
    // Uncomment to help debugging failing tests
    // Logger.setClassLogLevels(Map(compiler.getClass.getName -> LogLevel.Debug))
    val input =
     """circuit Top :
        |  module Top :
        |    input clk: Clock
        |    input in: UInt<3>
        |    mem m:
        |      data-type => {a: UInt<4>, b: UInt<4>[2]}
        |      depth => 8
        |      write-latency => 1
        |      read-latency => 0
        |      reader => r
        |    m.r.clk <= clk
        |    m.r.en <= UInt(1)
        |    m.r.addr <= in
        |""".stripMargin
    val annos = Seq(anno("m.r.data.b", "sub"), anno("m.r.data", "all"), anno("m", "mem"),
                    dontTouch("Top.m"))
    val result = compiler.compile(CircuitState(parse(input), ChirrtlForm, annos), Nil)
    val resultAnno = result.annotations.toSeq
    resultAnno should contain (anno("m_a", "mem"))
    resultAnno should contain (anno("m_b_0", "mem"))
    resultAnno should contain (anno("m_b_1", "mem"))
    resultAnno should contain (anno("m_a.r.data", "all"))
    resultAnno should contain (anno("m_b_0.r.data", "all"))
    resultAnno should contain (anno("m_b_1.r.data", "all"))
    resultAnno should contain (anno("m_b_0.r.data", "sub"))
    resultAnno should contain (anno("m_b_1.r.data", "sub"))
    resultAnno should not contain (anno("m"))
    resultAnno should not contain (anno("r"))
  }
  "Renaming" should "propagate in RemoveChirrtl and Lowering of memories" in {
    val compiler = new VerilogCompiler
    val input =
     """circuit Top :
        |  module Top :
        |    input clk: Clock
        |    input in: UInt<3>
        |    cmem m: {a: UInt<4>, b: UInt<4>[2]}[8]
        |    read mport r = m[in], clk
        |""".stripMargin
    val annos = Seq(anno("r.b", "sub"), anno("r", "all"), anno("m", "mem"), dontTouch("Top.m"))
    val result = compiler.compile(CircuitState(parse(input), ChirrtlForm, annos), Nil)
    val resultAnno = result.annotations.toSeq
    resultAnno should contain (anno("m_a", "mem"))
    resultAnno should contain (anno("m_b_0", "mem"))
    resultAnno should contain (anno("m_b_1", "mem"))
    resultAnno should contain (anno("m_a.r.data", "all"))
    resultAnno should contain (anno("m_b_0.r.data", "all"))
    resultAnno should contain (anno("m_b_1.r.data", "all"))
    resultAnno should contain (anno("m_b_0.r.data", "sub"))
    resultAnno should contain (anno("m_b_1.r.data", "sub"))
    resultAnno should not contain (anno("m"))
    resultAnno should not contain (anno("r"))
  }

  "Renaming" should "propagate in ZeroWidth" in {
    val compiler = new VerilogCompiler
    val input =
     """circuit Top :
        |  module Top :
        |    input zero: UInt<0>
        |    wire x: {a: UInt<3>, b: UInt<0>}
        |    wire y: UInt<0>[3]
        |    y[0] <= zero
        |    y[1] <= zero
        |    y[2] <= zero
        |    x.a <= zero
        |    x.b <= zero
        |""".stripMargin
    val annos = Seq(anno("zero"), anno("x.a"), anno("x.b"), anno("y[0]"), anno("y[1]"),
                    anno("y[2]"), dontTouch("Top.x"))
    val result = compiler.compile(CircuitState(parse(input), ChirrtlForm, annos), Nil)
    val resultAnno = result.annotations.toSeq
    resultAnno should contain (anno("x_a"))
    resultAnno should not contain (anno("zero"))
    resultAnno should not contain (anno("x.a"))
    resultAnno should not contain (anno("x.b"))
    resultAnno should not contain (anno("x_b"))
    resultAnno should not contain (anno("y[0]"))
    resultAnno should not contain (anno("y[1]"))
    resultAnno should not contain (anno("y[2]"))
    resultAnno should not contain (anno("y_0"))
    resultAnno should not contain (anno("y_1"))
    resultAnno should not contain (anno("y_2"))
  }

  "Renaming subcomponents" should "propagate in Lowering" in {
    val compiler = new VerilogCompiler
    val input =
     """circuit Top :
        |  module Top :
        |    input clk: Clock
        |    input pred: UInt<1>
        |    input in: {a: UInt<3>, b: UInt<3>[2]}
        |    output out: {a: UInt<3>, b: UInt<3>[2]}
        |    wire w: {a: UInt<3>, b: UInt<3>[2]}
        |    w is invalid
        |    out <= mux(pred, in, w)
        |    reg r: {a: UInt<3>, b: UInt<3>[2]}, clk
        |    cmem mem: {a: UInt<3>, b: UInt<3>[2]}[8]
        |    write mport write = mem[pred], clk
        |    write <= in
        |""".stripMargin
    val annos = Seq(
      anno("in.a"), anno("in.b[0]"), anno("in.b[1]"),
      anno("out.a"), anno("out.b[0]"), anno("out.b[1]"),
      anno("w.a"), anno("w.b[0]"), anno("w.b[1]"),
      anno("r.a"), anno("r.b[0]"), anno("r.b[1]"),
      anno("write.a"), anno("write.b[0]"), anno("write.b[1]"),
      dontTouch("Top.r"), dontTouch("Top.w"), dontTouch("Top.mem")
    )
    val result = compiler.compile(CircuitState(parse(input), ChirrtlForm, annos), Nil)
    val resultAnno = result.annotations.toSeq
    resultAnno should not contain (anno("in.a"))
    resultAnno should not contain (anno("in.b[0]"))
    resultAnno should not contain (anno("in.b[1]"))
    resultAnno should not contain (anno("out.a"))
    resultAnno should not contain (anno("out.b[0]"))
    resultAnno should not contain (anno("out.b[1]"))
    resultAnno should not contain (anno("w.a"))
    resultAnno should not contain (anno("w.b[0]"))
    resultAnno should not contain (anno("w.b[1]"))
    resultAnno should not contain (anno("n.a"))
    resultAnno should not contain (anno("n.b[0]"))
    resultAnno should not contain (anno("n.b[1]"))
    resultAnno should not contain (anno("r.a"))
    resultAnno should not contain (anno("r.b[0]"))
    resultAnno should not contain (anno("r.b[1]"))
    resultAnno should contain (anno("in_a"))
    resultAnno should contain (anno("in_b_0"))
    resultAnno should contain (anno("in_b_1"))
    resultAnno should contain (anno("out_a"))
    resultAnno should contain (anno("out_b_0"))
    resultAnno should contain (anno("out_b_1"))
    resultAnno should contain (anno("w_a"))
    resultAnno should contain (anno("w_b_0"))
    resultAnno should contain (anno("w_b_1"))
    resultAnno should contain (anno("r_a"))
    resultAnno should contain (anno("r_b_0"))
    resultAnno should contain (anno("r_b_1"))
    resultAnno should contain (anno("mem_a.write.data"))
    resultAnno should contain (anno("mem_b_0.write.data"))
    resultAnno should contain (anno("mem_b_1.write.data"))
  }

  "Renaming components" should "expand in Lowering" in {
    val compiler = new VerilogCompiler
    val input =
     """circuit Top :
        |  module Top :
        |    input clk: Clock
        |    input pred: UInt<1>
        |    input in: {a: UInt<3>, b: UInt<3>[2]}
        |    output out: {a: UInt<3>, b: UInt<3>[2]}
        |    wire w: {a: UInt<3>, b: UInt<3>[2]}
        |    w is invalid
        |    out <= mux(pred, in, w)
        |    reg r: {a: UInt<3>, b: UInt<3>[2]}, clk
        |""".stripMargin
    val annos = Seq(anno("in"), anno("out"), anno("w"), anno("r"), dontTouch("Top.r"),
                    dontTouch("Top.w"))
    val result = compiler.compile(CircuitState(parse(input), ChirrtlForm, annos), Nil)
    val resultAnno = result.annotations.toSeq
    resultAnno should contain (anno("in_a"))
    resultAnno should contain (anno("in_b_0"))
    resultAnno should contain (anno("in_b_1"))
    resultAnno should contain (anno("out_a"))
    resultAnno should contain (anno("out_b_0"))
    resultAnno should contain (anno("out_b_1"))
    resultAnno should contain (anno("w_a"))
    resultAnno should contain (anno("w_b_0"))
    resultAnno should contain (anno("w_b_1"))
    resultAnno should contain (anno("r_a"))
    resultAnno should contain (anno("r_b_0"))
    resultAnno should contain (anno("r_b_1"))
  }

  "Renaming subcomponents that aren't leaves" should "expand in Lowering" in {
    val compiler = new VerilogCompiler
    val input =
     """circuit Top :
        |  module Top :
        |    input clk: Clock
        |    input pred: UInt<1>
        |    input in: {a: UInt<3>, b: UInt<3>[2]}
        |    output out: {a: UInt<3>, b: UInt<3>[2]}
        |    wire w: {a: UInt<3>, b: UInt<3>[2]}
        |    w is invalid
        |    node n = mux(pred, in, w)
        |    out <= n
        |    reg r: {a: UInt<3>, b: UInt<3>[2]}, clk
        |""".stripMargin
    val annos = Seq(anno("in.b"), anno("out.b"), anno("w.b"), anno("r.b"),
                    dontTouch("Top.r"), dontTouch("Top.w"))
    val result = compiler.compile(CircuitState(parse(input), ChirrtlForm, annos), Nil)
    val resultAnno = result.annotations.toSeq
    resultAnno should contain (anno("in_b_0"))
    resultAnno should contain (anno("in_b_1"))
    resultAnno should contain (anno("out_b_0"))
    resultAnno should contain (anno("out_b_1"))
    resultAnno should contain (anno("w_b_0"))
    resultAnno should contain (anno("w_b_1"))
    resultAnno should contain (anno("r_b_0"))
    resultAnno should contain (anno("r_b_1"))
  }

  "Renaming" should "track constprop + dce" in {
    val compiler = new VerilogCompiler
    val input =
     """circuit Top :
        |  module Top :
        |    input clk: Clock
        |    input pred: UInt<1>
        |    input in: {a: UInt<3>, b: UInt<3>[2]}
        |    output out: {a: UInt<3>, b: UInt<3>[2]}
        |    node n = in
        |    out <= n
        |""".stripMargin
    val annos = Seq(
      anno("in.a"), anno("in.b[0]"), anno("in.b[1]"),
      anno("out.a"), anno("out.b[0]"), anno("out.b[1]"),
      anno("n.a"), anno("n.b[0]"), anno("n.b[1]")
    )
    val result = compiler.compile(CircuitState(parse(input), ChirrtlForm, annos), Nil)
    val resultAnno = result.annotations.toSeq
    resultAnno should not contain (anno("in.a"))
    resultAnno should not contain (anno("in.b[0]"))
    resultAnno should not contain (anno("in.b[1]"))
    resultAnno should not contain (anno("out.a"))
    resultAnno should not contain (anno("out.b[0]"))
    resultAnno should not contain (anno("out.b[1]"))
    resultAnno should not contain (anno("n.a"))
    resultAnno should not contain (anno("n.b[0]"))
    resultAnno should not contain (anno("n.b[1]"))
    resultAnno should not contain (anno("n_a"))
    resultAnno should not contain (anno("n_b_0"))
    resultAnno should not contain (anno("n_b_1"))
    resultAnno should contain (anno("in_a"))
    resultAnno should contain (anno("in_b_0"))
    resultAnno should contain (anno("in_b_1"))
    resultAnno should contain (anno("out_a"))
    resultAnno should contain (anno("out_b_0"))
    resultAnno should contain (anno("out_b_1"))
  }

  "Renaming" should "track deleted modules AND instances in dce" in {
    val compiler = new VerilogCompiler
    val input =
     """circuit Top :
        |  module Dead :
        |    input foo : UInt<8>
        |    output bar : UInt<8>
        |    bar <= foo
        |  extmodule DeadExt :
        |    input foo : UInt<8>
        |    output bar : UInt<8>
        |  module Top :
        |    input foo : UInt<8>
        |    output bar : UInt<8>
        |    inst d of Dead
        |    d.foo <= foo
        |    inst d2 of DeadExt
        |    d2.foo <= foo
        |    bar <= foo
        |""".stripMargin
    val annos = Seq(
      OptimizableExtModuleAnnotation(ModuleName("DeadExt", CircuitName("Top"))),
      manno("Dead"), manno("DeadExt"), manno("Top"),
      anno("d"), anno("d2"),
      anno("foo", mod = "Top"), anno("bar", mod = "Top"),
      anno("foo", mod = "Dead"), anno("bar", mod = "Dead"),
      anno("foo", mod = "DeadExt"), anno("bar", mod = "DeadExt")
    )
    val result = compiler.compile(CircuitState(parse(input), ChirrtlForm, annos), Nil)
    /* Uncomment to help debug
    println(result.circuit.serialize)
    result.annotations.foreach{ a =>
      a match {
        case DeletedAnnotation(xform, anno) => println(s"$xform deleted: ${a.target}")
        case Annotation(target, _, _) => println(s"not deleted: $target")
      }
    }
    */
    val resultAnno = result.annotations.toSeq

    resultAnno should contain (manno("Top"))
    resultAnno should contain (anno("foo", mod = "Top"))
    resultAnno should contain (anno("bar", mod = "Top"))

    resultAnno should not contain (manno("Dead"))
    resultAnno should not contain (manno("DeadExt"))
    resultAnno should not contain (anno("d"))
    resultAnno should not contain (anno("d2"))
    resultAnno should not contain (anno("foo", mod = "Dead"))
    resultAnno should not contain (anno("bar", mod = "Dead"))
    resultAnno should not contain (anno("foo", mod = "DeadExt"))
    resultAnno should not contain (anno("bar", mod = "DeadExt"))
  }

  "Renaming" should "track deduplication" in {
    val compiler = new VerilogCompiler
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
      anno("x", mod = "Child"), anno("y", mod = "Child_1"), manno("Child"), manno("Child_1")
    )
    val result = compiler.compile(CircuitState(parse(input), ChirrtlForm, annos), Nil)
    val resultAnno = result.annotations.toSeq
    resultAnno should contain (anno("x", mod = "Child"))
    resultAnno should contain (anno("y", mod = "Child"))
    resultAnno should contain (manno("Child"))
    resultAnno should not contain (anno("y", mod = "Child_1"))
    resultAnno should not contain (manno("Child_1"))
  }

  "AnnotationUtils.toNamed" should "invert Named.serialize" in {
    val x = ComponentName("component", ModuleName("module", CircuitName("circuit")))
    val y = AnnotationUtils.toNamed(x.serialize)
    require(x == y)
  }

  "Annotations on empty aggregates" should "be deleted" in {
    val compiler = new VerilogCompiler
    val input =
     """circuit Top :
        |  module Top :
        |    input x : { foo : UInt<8>, bar : {}, fizz : UInt<8>[0], buzz : UInt<0> }
        |    output y : { foo : UInt<8>, bar : {}, fizz : UInt<8>[0], buzz : UInt<0> }
        |    output a : {}
        |    output b : UInt<8>[0]
        |    output c : { d : UInt<0>, e : UInt<8> }[2]
        |    c is invalid
        |    y <= x
        |""".stripMargin
    val annos = Seq(
      anno("x"), anno("y.bar"), anno("y.fizz"), anno("y.buzz"), anno("a"), anno("b"), anno("c"),
      anno("c[0].d"), anno("c[1].d")
    )
    val result = compiler.compile(CircuitState(parse(input), ChirrtlForm, annos), Nil)
    val resultAnno = result.annotations.toSeq
    resultAnno should contain (anno("x_foo"))
    resultAnno should not contain (anno("a"))
    resultAnno should not contain (anno("b"))
    // Check both with and without dots because both are wrong
    resultAnno should not contain (anno("y.bar"))
    resultAnno should not contain (anno("y.fizz"))
    resultAnno should not contain (anno("y.buzz"))
    resultAnno should not contain (anno("x.bar"))
    resultAnno should not contain (anno("x.fizz"))
    resultAnno should not contain (anno("x.buzz"))
    resultAnno should not contain (anno("y_bar"))
    resultAnno should not contain (anno("y_fizz"))
    resultAnno should not contain (anno("y_buzz"))
    resultAnno should not contain (anno("x_bar"))
    resultAnno should not contain (anno("x_fizz"))
    resultAnno should not contain (anno("x_buzz"))
    resultAnno should not contain (anno("c"))
    resultAnno should contain (anno("c_0_e"))
    resultAnno should contain (anno("c_1_e"))
    resultAnno should not contain (anno("c[0].d"))
    resultAnno should not contain (anno("c[1].d"))
    resultAnno should not contain (anno("c_0_d"))
    resultAnno should not contain (anno("c_1_d"))
  }
}

class LegacyAnnotationTests extends AnnotationTests {
  def anno(s: String, value: String ="this is a value", mod: String = "Top"): Annotation =
    Annotation(ComponentName(s, ModuleName(mod, CircuitName("Top"))), classOf[Transform], value)
  def manno(mod: String): Annotation =
    Annotation(ModuleName(mod, CircuitName("Top")), classOf[Transform], "some value")

  "LegacyAnnotations" should "be readable from file" in {
    val annotationsYaml = FileUtils.getTextResource("/annotations/SampleAnnotations.anno").parseYaml
    val annotationArray = annotationsYaml.convertTo[Array[LegacyAnnotation]]
    annotationArray.length should be (9)
    annotationArray(0).targetString should be ("ModC")
    annotationArray(7).transformClass should be ("firrtl.passes.InlineInstances")
    val expectedValue = "TopOfDiamond\nWith\nSome new lines"
    annotationArray(7).value should be (expectedValue)
  }

  "Badly formatted LegacyAnnotation serializations" should "return reasonable error messages" in {
    var badYaml =
      """
        |- transformClass: firrtl.passes.InlineInstances
        |  targetString: circuit.module..
        |  value: ModC.this params 16 32
      """.stripMargin.parseYaml

    var thrown = intercept[Exception] {
      badYaml.convertTo[Array[LegacyAnnotation]]
    }
    thrown.getMessage should include ("Illegal component name")

    badYaml =
      """
        |- transformClass: firrtl.passes.InlineInstances
        |  targetString: .circuit.module.component
        |  value: ModC.this params 16 32
      """.stripMargin.parseYaml

    thrown = intercept[Exception] {
      badYaml.convertTo[Array[LegacyAnnotation]]
    }
    thrown.getMessage should include ("Illegal circuit name")
  }

}

class JsonAnnotationTests extends AnnotationTests with BackendCompilationUtilities {
  // Helper annotations
  case class SimpleAnno(target: ComponentName, value: String) extends
      SingleTargetAnnotation[ComponentName] {
    def duplicate(n: ComponentName) = this.copy(target = n)
  }
  case class ModuleAnno(target: ModuleName) extends SingleTargetAnnotation[ModuleName] {
    def duplicate(n: ModuleName) = this.copy(target = n)
  }

  def anno(s: String, value: String ="this is a value", mod: String = "Top"): SimpleAnno =
    SimpleAnno(ComponentName(s, ModuleName(mod, CircuitName("Top"))), value)
  def manno(mod: String): Annotation = ModuleAnno(ModuleName(mod, CircuitName("Top")))

  "Round tripping annotations through text file" should "preserve annotations" in {
    val annos: Array[Annotation] = Seq(
      InlineAnnotation(CircuitName("fox")),
      InlineAnnotation(ModuleName("dog", CircuitName("bear"))),
      InlineAnnotation(ComponentName("chocolate", ModuleName("like", CircuitName("i")))),
      InlineAnnotation(ComponentName("chocolate.frog", ModuleName("like", CircuitName("i")))),
      PinAnnotation(Seq("sea-lion", "monk-seal"))
    ).toArray

    val annoFile = new File("temp-anno")
    val writer = new FileWriter(annoFile)
    writer.write(JsonProtocol.serialize(annos))
    writer.close()

    val text = FileUtils.getText(annoFile)
    annoFile.delete()

    val readAnnos = JsonProtocol.deserializeTry(text).get

    annos should be (readAnnos)
  }

  private def setupManager(annoFileText: Option[String]) = {
    val source = """
      |circuit test :
      |  module test :
      |    input x : UInt<1>
      |    output z : UInt<1>
      |    z <= x
      |    node y = x""".stripMargin
    val testDir = createTestDirectory(this.getClass.getSimpleName)
    val annoFile = new File(testDir, "anno.json")

    annoFileText.foreach { text =>
      val w = new FileWriter(annoFile)
      w.write(text)
      w.close()
    }

    new ExecutionOptionsManager("annos") with HasFirrtlOptions {
      commonOptions = CommonOptions(targetDirName = testDir.getPath)
      firrtlOptions = FirrtlExecutionOptions(
        firrtlSource = Some(source),
        annotationFileNames = List(annoFile.getPath)
      )
    }
  }

  "Annotation file not found" should "give a reasonable error message" in {
    val manager = setupManager(None)

    an [AnnotationFileNotFoundException] shouldBe thrownBy {
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

    the [Exception] thrownBy Driver.execute(manager) should matchPattern {
      case InvalidAnnotationFileException(_, _: AnnotationClassNotFoundException) =>
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

    the [Exception] thrownBy Driver.execute(manager) should matchPattern {
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

    the [Exception] thrownBy Driver.execute(manager) should matchPattern {
      case InvalidAnnotationFileException(_, InvalidAnnotationJSONException(msg))
        if msg.contains("JObject") =>
    }
  }

  object DoNothingTransform extends Transform {
    override def inputForm: CircuitForm = UnknownForm
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
}
