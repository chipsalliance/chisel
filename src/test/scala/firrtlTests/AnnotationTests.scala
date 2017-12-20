// See LICENSE for license details.

package firrtlTests

import java.io.{File, FileWriter, Writer}

import firrtl.annotations.AnnotationYamlProtocol._
import firrtl.annotations._
import firrtl._
import firrtl.transforms.OptimizableExtModuleAnnotation
import firrtl.passes.InlineAnnotation
import firrtl.passes.memlib.PinAnnotation
import firrtl.transforms.DontTouchAnnotation
import net.jcazevedo.moultingyaml._
import org.scalatest.Matchers
import logger._

/**
 * An example methodology for testing Firrtl annotations.
 */
trait AnnotationSpec extends LowTransformSpec {
  // Dummy transform
  def transform = new ResolveAndCheck

  // Check if Annotation Exception is thrown
  override def failingexecute(input: String, annotations: Seq[Annotation]): Exception = {
    intercept[AnnotationException] {
      compile(CircuitState(parse(input), ChirrtlForm, Some(AnnotationMap(annotations))), Seq.empty)
    }
  }
  def execute(input: String, check: Annotation, annotations: Seq[Annotation]): Unit = {
    val cr = compile(CircuitState(parse(input), ChirrtlForm, Some(AnnotationMap(annotations))), Seq.empty)
    cr.annotations.get.annotations should contain (check)
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
  def getAMap(a: Annotation): Option[AnnotationMap] = Some(AnnotationMap(Seq(a)))
  def getAMap(as: Seq[Annotation]): Option[AnnotationMap] = Some(AnnotationMap(as))
  def anno(s: String, value: String ="this is a value", mod: String = "Top"): Annotation =
    Annotation(ComponentName(s, ModuleName(mod, CircuitName("Top"))), classOf[Transform], value)
  def manno(mod: String): Annotation =
    Annotation(ModuleName(mod, CircuitName("Top")), classOf[Transform], "some value")

  "Loose and Sticky annotation on a node" should "pass through" in {
    val input: String =
      """circuit Top :
         |  module Top :
         |    input a : UInt<1>[2]
         |    input b : UInt<1>
         |    node c = b""".stripMargin
    val ta = anno("c", "")
    execute(input, ta, Seq(ta))
  }

  "Annotations" should "be readable from file" in {
    val annotationStream = getClass.getResourceAsStream("/annotations/SampleAnnotations.anno")
    val annotationsYaml = scala.io.Source.fromInputStream(annotationStream).getLines().mkString("\n").parseYaml
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
      def execute(state: CircuitState) = state.copy(annotations = None)
    }
    val inlineAnn = InlineAnnotation(CircuitName("Top"))
    val result = compiler.compile(CircuitState(parse(input), ChirrtlForm, getAMap(inlineAnn)), Seq(new DeletingTransform))
    result.annotations.get.annotations.head should matchPattern {
      case DeletedAnnotation(x, inlineAnn) =>
    }
    val exception = (intercept[FIRRTLException] {
      result.getEmittedCircuit
    })
    val deleted = result.deletedAnnotations
    exception.str should be (s"No EmittedCircuit found! Did you delete any annotations?\n$deleted")
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
    val result = compiler.compile(CircuitState(parse(input), ChirrtlForm, getAMap(annos)), Nil)
    val resultAnno = result.annotations.get.annotations
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
    Logger.setClassLogLevels(Map(compiler.getClass.getName -> LogLevel.Debug))
    val input =
     """circuit Top :
        |  module Top :
        |    input clk: Clock
        |    input in: UInt<3>
        |    cmem m: {a: UInt<4>, b: UInt<4>[2]}[8]
        |    read mport r = m[in], clk
        |""".stripMargin
    val annos = Seq(anno("r.b", "sub"), anno("r", "all"), anno("m", "mem"), dontTouch("Top.m"))
    val result = compiler.compile(CircuitState(parse(input), ChirrtlForm, getAMap(annos)), Nil)
    val resultAnno = result.annotations.get.annotations
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
    val result = compiler.compile(CircuitState(parse(input), ChirrtlForm, getAMap(annos)), Nil)
    val resultAnno = result.annotations.get.annotations
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
        |    node n = mux(pred, in, w)
        |    out <= n
        |    reg r: {a: UInt<3>, b: UInt<3>[2]}, clk
        |    cmem mem: {a: UInt<3>, b: UInt<3>[2]}[8]
        |    write mport write = mem[pred], clk
        |    write <= in
        |""".stripMargin
    val annos = Seq(
      anno("in.a"), anno("in.b[0]"), anno("in.b[1]"),
      anno("out.a"), anno("out.b[0]"), anno("out.b[1]"),
      anno("w.a"), anno("w.b[0]"), anno("w.b[1]"),
      anno("n.a"), anno("n.b[0]"), anno("n.b[1]"),
      anno("r.a"), anno("r.b[0]"), anno("r.b[1]"),
      anno("write.a"), anno("write.b[0]"), anno("write.b[1]"),
      dontTouch("Top.r"), dontTouch("Top.w")
    )
    val result = compiler.compile(CircuitState(parse(input), ChirrtlForm, getAMap(annos)), Nil)
    val resultAnno = result.annotations.get.annotations
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
    resultAnno should contain (anno("n_a"))
    resultAnno should contain (anno("n_b_0"))
    resultAnno should contain (anno("n_b_1"))
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
        |    node n = mux(pred, in, w)
        |    out <= n
        |    reg r: {a: UInt<3>, b: UInt<3>[2]}, clk
        |""".stripMargin
    val annos = Seq(anno("in"), anno("out"), anno("w"), anno("n"), anno("r"), dontTouch("Top.r"),
                    dontTouch("Top.w"))
    val result = compiler.compile(CircuitState(parse(input), ChirrtlForm, getAMap(annos)), Nil)
    val resultAnno = result.annotations.get.annotations
    resultAnno should contain (anno("in_a"))
    resultAnno should contain (anno("in_b_0"))
    resultAnno should contain (anno("in_b_1"))
    resultAnno should contain (anno("out_a"))
    resultAnno should contain (anno("out_b_0"))
    resultAnno should contain (anno("out_b_1"))
    resultAnno should contain (anno("w_a"))
    resultAnno should contain (anno("w_b_0"))
    resultAnno should contain (anno("w_b_1"))
    resultAnno should contain (anno("n_a"))
    resultAnno should contain (anno("n_b_0"))
    resultAnno should contain (anno("n_b_1"))
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
    val annos = Seq(anno("in.b"), anno("out.b"), anno("w.b"), anno("n.b"), anno("r.b"),
                    dontTouch("Top.r"), dontTouch("Top.w"))
    val result = compiler.compile(CircuitState(parse(input), ChirrtlForm, getAMap(annos)), Nil)
    val resultAnno = result.annotations.get.annotations
    resultAnno should contain (anno("in_b_0"))
    resultAnno should contain (anno("in_b_1"))
    resultAnno should contain (anno("out_b_0"))
    resultAnno should contain (anno("out_b_1"))
    resultAnno should contain (anno("w_b_0"))
    resultAnno should contain (anno("w_b_1"))
    resultAnno should contain (anno("n_b_0"))
    resultAnno should contain (anno("n_b_1"))
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
    val result = compiler.compile(CircuitState(parse(input), ChirrtlForm, getAMap(annos)), Nil)
    val resultAnno = result.annotations.get.annotations
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
    val result = compiler.compile(CircuitState(parse(input), ChirrtlForm, getAMap(annos)), Nil)
    /* Uncomment to help debug
    println(result.circuit.serialize)
    result.annotations.get.annotations.foreach{ a =>
      a match {
        case DeletedAnnotation(xform, anno) => println(s"$xform deleted: ${a.target}")
        case Annotation(target, _, _) => println(s"not deleted: $target")
      }
    }
    */
    val resultAnno = result.annotations.get.annotations

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
    val result = compiler.compile(CircuitState(parse(input), ChirrtlForm, getAMap(annos)), Nil)
    val resultAnno = result.annotations.get.annotations
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
}
