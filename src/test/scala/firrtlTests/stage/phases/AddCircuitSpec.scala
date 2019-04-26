// See LICENSE for license details.

package firrtlTests.stage.phases

import org.scalatest.{FlatSpec, Matchers}

import firrtl.{AnnotationSeq, Parser}
import firrtl.annotations.NoTargetAnnotation
import firrtl.options.{OptionsException, Phase, PhasePrerequisiteException}
import firrtl.stage.{CircuitOption, FirrtlCircuitAnnotation, FirrtlSourceAnnotation, InfoModeAnnotation,
  FirrtlFileAnnotation}
import firrtl.stage.phases.AddCircuit

import java.io.{File, FileWriter}

class AddCircuitSpec extends FlatSpec with Matchers {

  case class FooAnnotation(x: Int) extends NoTargetAnnotation
  case class BarAnnotation(x: String) extends NoTargetAnnotation

  class Fixture { val phase: Phase = new AddCircuit }

  behavior of classOf[AddCircuit].toString

  def firrtlSource(name: String): String =
    s"""|circuit $name:
        |  module $name:
        |    input a: UInt<1>
        |    output b: UInt<1>
        |    b <= not(a)
        |""".stripMargin

  it should "throw a PhasePrerequisiteException if a CircuitOption exists without an InfoModeAnnotation" in
  new Fixture {
    {the [PhasePrerequisiteException] thrownBy phase.transform(Seq(FirrtlSourceAnnotation("foo")))}
      .message should startWith ("An InfoModeAnnotation must be present")
  }

  it should "do nothing if no CircuitOption annotations are present" in new Fixture {
    val annotations = (1 to 10).map(FooAnnotation) ++
      ('a' to 'm').map(_.toString).map(BarAnnotation) :+ InfoModeAnnotation("ignore")
    phase.transform(annotations).toSeq should be (annotations.toSeq)
  }

  val (file, fileCircuit) = {
    val source = firrtlSource("foo")
    val fileName = "test_run_dir/AddCircuitSpec.fir"
    val fw = new FileWriter(new File(fileName))
    fw.write(source)
    fw.close()
    (fileName, Parser.parse(source))
  }

  val (source, sourceCircuit) = {
    val source = firrtlSource("bar")
    (source, Parser.parse(source))
  }

  it should "transform and remove CircuitOption annotations" in new Fixture {
    val circuit = Parser.parse(firrtlSource("baz"))

    val annotations = Seq(
      FirrtlFileAnnotation(file),
      FirrtlSourceAnnotation(source),
      FirrtlCircuitAnnotation(circuit),
      InfoModeAnnotation("ignore") )

    val annotationsExpected = Set(
      FirrtlCircuitAnnotation(fileCircuit),
      FirrtlCircuitAnnotation(sourceCircuit),
      FirrtlCircuitAnnotation(circuit) )

    val out = phase.transform(annotations).toSeq

    info("generated expected FirrtlCircuitAnnotations")
    out.collect{ case a: FirrtlCircuitAnnotation => a}.toSet should be (annotationsExpected)

    info("all CircuitOptions were removed")
    out.collect{ case a: CircuitOption => a } should be (empty)
  }

  it should """add info for a FirrtlFileAnnotation with a "gen" info mode""" in new Fixture {
    phase.transform(Seq(InfoModeAnnotation("gen"), FirrtlFileAnnotation(file)))
      .collectFirst{ case a: FirrtlCircuitAnnotation => a.circuit.serialize }
      .get should include ("AddCircuitSpec")
  }

  it should """add info for a FirrtlSourceAnnotation with an "append" info mode""" in new Fixture {
    phase.transform(Seq(InfoModeAnnotation("append"), FirrtlSourceAnnotation(source)))
      .collectFirst{ case a: FirrtlCircuitAnnotation => a.circuit.serialize }
      .get should include ("anonymous source")
  }

  it should "throw an OptionsException if the specified file doesn't exist" in new Fixture {
    val a = Seq(InfoModeAnnotation("ignore"), FirrtlFileAnnotation("test_run_dir/I-DO-NOT-EXIST"))

    {the [OptionsException] thrownBy phase.transform(a)}
      .message should startWith (s"Input file 'test_run_dir/I-DO-NOT-EXIST' not found")
  }

}
