// SPDX-License-Identifier: Apache-2.0

package firrtlTests

import java.io.{File, FileWriter}
import firrtl.annotations._
import firrtl._
import firrtl.FileUtils
import firrtl.options.{Dependency, InputAnnotationFileAnnotation}
import firrtl.transforms.OptimizableExtModuleAnnotation
import firrtl.passes.InlineAnnotation
import firrtl.passes.memlib.PinAnnotation
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

  "Deleting annotations" should "create a DeletedAnnotation" in {
    val transform = Dependency[DeletingTransform]
    val compiler = makeVerilogCompiler(Seq(transform))
    val input =
      """circuit Top :
        |  module Top :
        |    input in: UInt<3>
        |""".stripMargin

    val tname = transform.getName
    val inlineAnn = InlineAnnotation(CircuitName("Top"))
    val result = compiler.transform(CircuitState(parse(input), Seq(inlineAnn)))
    result.annotations.last should matchPattern {
      case DeletedAnnotation(`tname`, `inlineAnn`) =>
    }
    val exception = (intercept[Exception] {
      result.getEmittedCircuit
    })
    val deleted = result.deletedAnnotations
    exception.getMessage should be(s"No EmittedCircuit found! Did you delete any annotations?\n$deleted")
  }

  "Renaming" should "propagate in Lowering of memories" in {
    val compiler = makeVerilogCompiler()
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
    val annos = Seq(anno("m.r.data.b", "sub"), anno("m.r.data", "all"), anno("m", "mem"), dontTouch("Top.m"))
    val result = compiler.transform(CircuitState(parse(input), annos))
    val resultAnno = result.annotations.toSeq
    resultAnno should contain(anno("m_a", "mem"))
    resultAnno should contain(anno("m_b_0", "mem"))
    resultAnno should contain(anno("m_b_1", "mem"))
    resultAnno should contain(anno("m_a.r.data", "all"))
    resultAnno should contain(anno("m_b_0.r.data", "all"))
    resultAnno should contain(anno("m_b_1.r.data", "all"))
    resultAnno should contain(anno("m_b_0.r.data", "sub"))
    resultAnno should contain(anno("m_b_1.r.data", "sub"))
    resultAnno should not contain (anno("m"))
    resultAnno should not contain (anno("r"))
  }
  "Renaming" should "propagate in RemoveChirrtl and Lowering of memories" in {
    val compiler = makeVerilogCompiler()
    val input =
      """circuit Top :
        |  module Top :
        |    input clk: Clock
        |    input in: UInt<3>
        |    cmem m: {a: UInt<4>, b: UInt<4>[2]}[8]
        |    read mport r = m[in], clk
        |""".stripMargin
    val annos = Seq(anno("r.b", "sub"), anno("r", "all"), anno("m", "mem"), dontTouch("Top.m"))
    val result = compiler.transform(CircuitState(parse(input), annos))
    val resultAnno = result.annotations.toSeq
    resultAnno should contain(anno("m_a", "mem"))
    resultAnno should contain(anno("m_b_0", "mem"))
    resultAnno should contain(anno("m_b_1", "mem"))
    resultAnno should contain(anno("m_a.r.data", "all"))
    resultAnno should contain(anno("m_b_0.r.data", "all"))
    resultAnno should contain(anno("m_b_1.r.data", "all"))
    resultAnno should contain(anno("m_b_0.r.data", "sub"))
    resultAnno should contain(anno("m_b_1.r.data", "sub"))
    resultAnno should not contain (anno("m"))
    resultAnno should not contain (anno("r"))
  }

  "Renaming" should "propagate in ZeroWidth" in {
    val compiler = makeVerilogCompiler()
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
    val annos =
      Seq(anno("zero"), anno("x.a"), anno("x.b"), anno("y[0]"), anno("y[1]"), anno("y[2]"), dontTouch("Top.x"))
    val result = compiler.transform(CircuitState(parse(input), annos))
    val resultAnno = result.annotations.toSeq
    resultAnno should contain(anno("x_a"))
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
    val compiler = makeVerilogCompiler()
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
      anno("in.a"),
      anno("in.b[0]"),
      anno("in.b[1]"),
      anno("out.a"),
      anno("out.b[0]"),
      anno("out.b[1]"),
      anno("w.a"),
      anno("w.b[0]"),
      anno("w.b[1]"),
      anno("r.a"),
      anno("r.b[0]"),
      anno("r.b[1]"),
      anno("write.a"),
      anno("write.b[0]"),
      anno("write.b[1]"),
      dontTouch("Top.r"),
      dontTouch("Top.w"),
      dontTouch("Top.mem")
    )
    val result = compiler.transform(CircuitState(parse(input), annos))
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
    resultAnno should contain(anno("in_a"))
    resultAnno should contain(anno("in_b_0"))
    resultAnno should contain(anno("in_b_1"))
    resultAnno should contain(anno("out_a"))
    resultAnno should contain(anno("out_b_0"))
    resultAnno should contain(anno("out_b_1"))
    resultAnno should contain(anno("w_a"))
    resultAnno should contain(anno("w_b_0"))
    resultAnno should contain(anno("w_b_1"))
    resultAnno should contain(anno("r_a"))
    resultAnno should contain(anno("r_b_0"))
    resultAnno should contain(anno("r_b_1"))
    resultAnno should contain(anno("mem_a.write.data"))
    resultAnno should contain(anno("mem_b_0.write.data"))
    resultAnno should contain(anno("mem_b_1.write.data"))
  }

  "Renaming components" should "expand in Lowering" in {
    val compiler = makeVerilogCompiler()
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
    val annos = Seq(anno("in"), anno("out"), anno("w"), anno("r"), dontTouch("Top.r"), dontTouch("Top.w"))
    val result = compiler.transform(CircuitState(parse(input), annos))
    val resultAnno = result.annotations.toSeq
    resultAnno should contain(anno("in_a"))
    resultAnno should contain(anno("in_b_0"))
    resultAnno should contain(anno("in_b_1"))
    resultAnno should contain(anno("out_a"))
    resultAnno should contain(anno("out_b_0"))
    resultAnno should contain(anno("out_b_1"))
    resultAnno should contain(anno("w_a"))
    resultAnno should contain(anno("w_b_0"))
    resultAnno should contain(anno("w_b_1"))
    resultAnno should contain(anno("r_a"))
    resultAnno should contain(anno("r_b_0"))
    resultAnno should contain(anno("r_b_1"))
  }

  "Renaming subcomponents that aren't leaves" should "expand in Lowering" in {
    val compiler = makeVerilogCompiler()
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
    val annos = Seq(anno("in.b"), anno("out.b"), anno("w.b"), anno("r.b"), dontTouch("Top.r"), dontTouch("Top.w"))
    val result = compiler.transform(CircuitState(parse(input), annos))
    val resultAnno = result.annotations.toSeq
    resultAnno should contain(anno("in_b_0"))
    resultAnno should contain(anno("in_b_1"))
    resultAnno should contain(anno("out_b_0"))
    resultAnno should contain(anno("out_b_1"))
    resultAnno should contain(anno("w_b_0"))
    resultAnno should contain(anno("w_b_1"))
    resultAnno should contain(anno("r_b_0"))
    resultAnno should contain(anno("r_b_1"))
  }

  "Renaming" should "track constprop + dce" in {
    val compiler = makeVerilogCompiler()
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
      anno("in.a"),
      anno("in.b[0]"),
      anno("in.b[1]"),
      anno("out.a"),
      anno("out.b[0]"),
      anno("out.b[1]"),
      anno("n.a"),
      anno("n.b[0]"),
      anno("n.b[1]")
    )
    val result = compiler.transform(CircuitState(parse(input), annos))
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
    resultAnno should contain(anno("in_a"))
    resultAnno should contain(anno("in_b_0"))
    resultAnno should contain(anno("in_b_1"))
    resultAnno should contain(anno("out_a"))
    resultAnno should contain(anno("out_b_0"))
    resultAnno should contain(anno("out_b_1"))
  }

  "Renaming" should "track deleted modules AND instances in dce" in {
    val compiler = makeVerilogCompiler()
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
      manno("Dead"),
      manno("DeadExt"),
      manno("Top"),
      anno("d"),
      anno("d2"),
      anno("foo", mod = "Top"),
      anno("bar", mod = "Top"),
      anno("foo", mod = "Dead"),
      anno("bar", mod = "Dead"),
      anno("foo", mod = "DeadExt"),
      anno("bar", mod = "DeadExt")
    )
    val result = compiler.transform(CircuitState(parse(input), annos))
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

    resultAnno should contain(manno("Top"))
    resultAnno should contain(anno("foo", mod = "Top"))
    resultAnno should contain(anno("bar", mod = "Top"))

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

  "Annotations on empty aggregates" should "be deleted" in {
    val compiler = makeVerilogCompiler()
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
      anno("x"),
      anno("y.bar"),
      anno("y.fizz"),
      anno("y.buzz"),
      anno("a"),
      anno("b"),
      anno("c"),
      anno("c[0].d"),
      anno("c[1].d")
    )
    val result = compiler.transform(CircuitState(parse(input), annos))
    val resultAnno = result.annotations.toSeq
    resultAnno should contain(anno("x_foo"))
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
    resultAnno should contain(anno("c_0_e"))
    resultAnno should contain(anno("c_1_e"))
    resultAnno should not contain (anno("c[0].d"))
    resultAnno should not contain (anno("c[1].d"))
    resultAnno should not contain (anno("c_0_d"))
    resultAnno should not contain (anno("c_1_d"))
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
