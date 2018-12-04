// See LICENSE for license details.

package firrtl.stage.phases.tests

import org.scalatest.{FlatSpec, Matchers, PrivateMethodTester}

import scala.io.Source

import java.io.File

import firrtl._
import firrtl.stage._
import firrtl.stage.phases.DriverCompatibility._

import firrtl.options.{InputAnnotationFileAnnotation, Phase, TargetDirAnnotation}
import firrtl.stage.phases.DriverCompatibility

class DriverCompatibilitySpec extends FlatSpec with Matchers with PrivateMethodTester {

  class PhaseFixture(val phase: Phase)

  /* This method wraps some magic that lets you use the private method DriverCompatibility.topName */
  def topName(annotations: AnnotationSeq): Option[String] = {
    val topName = PrivateMethod[Option[String]]('topName)
    DriverCompatibility invokePrivate topName(annotations)
  }

  def simpleCircuit(main: String): String = s"""|circuit $main:
                                                |  module $main:
                                                |    node x = UInt<1>("h0")
                                                |""".stripMargin

  /* This is a tuple holding an annotation that can be used to derive a top name and the expected top name for that
   * annotation. If these annotations are presented here in the same order that DriverCompatibility.topName uses to
   * discern a top name. E.g., a TopNameAnnotation is always used, even in the presence of a FirrtlSourceAnnotation.
   * Note: the last two FirrtlFileAnnotations have equal precedence, but the first one in the AnnotationSeq wins.
   */
  val annosWithTops = Seq(
    (TopNameAnnotation("foo"), "foo"),
    (FirrtlCircuitAnnotation(Parser.parse(simpleCircuit("bar"))), "bar"),
    (FirrtlSourceAnnotation(simpleCircuit("baz")), "baz"),
    (FirrtlFileAnnotation("src/test/resources/integration/PipeTester.fir"), "PipeTester"),
    (FirrtlFileAnnotation("src/test/resources/integration/GCDTester.pb"), "GCDTester")
  )

  behavior of s"${DriverCompatibility.getClass.getName}.topName (private method)"

  /* This iterates over the tails of annosWithTops. Using the ordering of annosWithTops, if this AnnotationSeq is fed to
   * DriverCompatibility.topName, the head annotation will be used to determine the top name. This test ensures that
   * topName behaves as expected.
   */
  for ( t <- annosWithTops.tails ) t match {
    case Nil =>
      it should "return None on an empty AnnotationSeq" in {
        topName(Seq.empty) should be (None)
      }
    case x =>
      val annotations = x.map(_._1)
      val top = x.head._2
      it should s"determine a top name ('$top') from a ${annotations.head.getClass.getName}" in {
        topName(annotations).get should be (top)
      }
  }

  def createFile(name: String): Unit = {
    val file = new File(name)
    file.getParentFile.getCanonicalFile.mkdirs()
    file.createNewFile()
  }

  behavior of classOf[AddImplicitAnnotationFile].toString

  val testDir = "test_run_dir/DriverCompatibilitySpec"

  it should "not modify the annotations if an InputAnnotationFile already exists" in
  new PhaseFixture(new AddImplicitAnnotationFile) {

    createFile(testDir + "/foo.anno")
    val annotations = Seq(
      InputAnnotationFileAnnotation("bar.anno"),
      TargetDirAnnotation(testDir),
      TopNameAnnotation("foo") )

    phase.transform(annotations).toSeq should be (annotations)
  }

  it should "add an InputAnnotationFile based on a derived topName" in
  new PhaseFixture(new AddImplicitAnnotationFile) {
    createFile(testDir + "/bar.anno")
    val annotations = Seq(
      TargetDirAnnotation(testDir),
      TopNameAnnotation("bar") )

    val expected = annotations.toSet +
      InputAnnotationFileAnnotation(testDir + "/bar.anno")

    phase.transform(annotations).toSet should be (expected)
  }

  it should "not add an InputAnnotationFile for .anno.json annotations" in
  new PhaseFixture(new AddImplicitAnnotationFile) {
    createFile(testDir + "/baz.anno.json")
    val annotations = Seq(
      TargetDirAnnotation(testDir),
      TopNameAnnotation("baz") )

    phase.transform(annotations).toSeq should be (annotations)
  }

  it should "not add an InputAnnotationFile if it cannot determine the topName" in
  new PhaseFixture(new AddImplicitAnnotationFile) {
    val annotations = Seq( TargetDirAnnotation(testDir) )

    phase.transform(annotations).toSeq should be (annotations)
  }

  behavior of classOf[AddImplicitFirrtlFile].toString

  it should "not modify the annotations if a CircuitOption is present" in
  new PhaseFixture(new AddImplicitFirrtlFile) {
    val annotations = Seq( FirrtlFileAnnotation("foo"), TopNameAnnotation("bar") )

    phase.transform(annotations).toSeq should be (annotations)
  }

  it should "add an FirrtlFileAnnotation if a TopNameAnnotation is present" in
  new PhaseFixture(new AddImplicitFirrtlFile) {
    val annotations = Seq( TopNameAnnotation("foo") )
    val expected = annotations.toSet +
      FirrtlFileAnnotation(new File("foo.fir").getCanonicalPath)

    phase.transform(annotations).toSet should be (expected)
  }

  it should "do nothing if no TopNameAnnotation is present" in
  new PhaseFixture(new AddImplicitFirrtlFile) {
    val annotations = Seq( TargetDirAnnotation("foo") )

    phase.transform(annotations).toSeq should be (annotations)
  }

  behavior of classOf[AddImplicitEmitter].toString

  val (nc, hfc, mfc, lfc, vc, svc) = ( new NoneCompiler,
                                       new HighFirrtlCompiler,
                                       new MiddleFirrtlCompiler,
                                       new LowFirrtlCompiler,
                                       new VerilogCompiler,
                                       new SystemVerilogCompiler )

  it should "convert CompilerAnnotations into EmitCircuitAnnotations without EmitOneFilePerModuleAnnotation" in
  new PhaseFixture(new AddImplicitEmitter) {
    val annotations = Seq(
      CompilerAnnotation(nc),
      CompilerAnnotation(hfc),
      CompilerAnnotation(mfc),
      CompilerAnnotation(lfc),
      CompilerAnnotation(vc),
      CompilerAnnotation(svc)
    )
    val expected = annotations
      .flatMap( a => Seq(a,
                         RunFirrtlTransformAnnotation(a.compiler.emitter),
                         EmitCircuitAnnotation(a.compiler.emitter.getClass)) )

    phase.transform(annotations).toSeq should be (expected)
  }

  it should "convert CompilerAnnotations into EmitAllodulesAnnotation with EmitOneFilePerModuleAnnotation" in
  new PhaseFixture(new AddImplicitEmitter) {
    val annotations = Seq(
      EmitOneFilePerModuleAnnotation,
      CompilerAnnotation(nc),
      CompilerAnnotation(hfc),
      CompilerAnnotation(mfc),
      CompilerAnnotation(lfc),
      CompilerAnnotation(vc),
      CompilerAnnotation(svc)
    )
    val expected = annotations
      .flatMap{
        case a: CompilerAnnotation => Seq(a,
                                          RunFirrtlTransformAnnotation(a.compiler.emitter),
                                          EmitAllModulesAnnotation(a.compiler.emitter.getClass))
        case a => Seq(a)
      }

    phase.transform(annotations).toSeq should be (expected)
  }

  behavior of classOf[AddImplicitOutputFile].toString

  it should "add an OutputFileAnnotation derived from a TopNameAnnotation if no OutputFileAnnotation exists" in
  new PhaseFixture(new AddImplicitOutputFile) {
    val annotations = Seq( TopNameAnnotation("foo") )
    val expected = Seq(
      OutputFileAnnotation("foo"),
      TopNameAnnotation("foo")
    )
    phase.transform(annotations).toSeq should be (expected)
  }

  it should "do nothing if an OutputFileannotation already exists" in
  new PhaseFixture(new AddImplicitOutputFile) {
    val annotations = Seq(
      TopNameAnnotation("foo"),
      OutputFileAnnotation("bar") )
    val expected = annotations
    phase.transform(annotations).toSeq should be (expected)
  }

  it should "do nothing if no TopNameAnnotation exists" in
  new PhaseFixture(new AddImplicitOutputFile) {
    val annotations = Seq.empty
    val expected = annotations
    phase.transform(annotations).toSeq should be (expected)
  }

}
