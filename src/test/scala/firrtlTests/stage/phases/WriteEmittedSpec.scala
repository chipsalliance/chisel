// See LICENSE for license details.

package firrtlTests.stage.phases

import org.scalatest.{FlatSpec, Matchers}

import java.io.File

import firrtl._

import firrtl.options.{Phase, TargetDirAnnotation}
import firrtl.stage.OutputFileAnnotation
import firrtl.stage.phases.WriteEmitted

class WriteEmittedSpec extends FlatSpec with Matchers {

  def removeEmitted(a: AnnotationSeq): AnnotationSeq = a.flatMap {
    case a: EmittedAnnotation[_] => None
    case a => Some(a)
  }

  class Fixture { val phase: Phase = new WriteEmitted }

  behavior of classOf[WriteEmitted].toString

  it should "write emitted circuits" in new Fixture {
    val annotations = Seq(
      TargetDirAnnotation("test_run_dir/WriteEmittedSpec"),
      EmittedFirrtlCircuitAnnotation(EmittedFirrtlCircuit("foo", "", ".foocircuit")),
      EmittedFirrtlCircuitAnnotation(EmittedFirrtlCircuit("bar", "", ".barcircuit")),
      EmittedVerilogCircuitAnnotation(EmittedVerilogCircuit("baz", "", ".bazcircuit")) )
    val expected = Seq("foo.foocircuit", "bar.barcircuit", "baz.bazcircuit")
      .map(a => new File(s"test_run_dir/WriteEmittedSpec/$a"))

    info("annotations are unmodified")
    phase.transform(annotations).toSeq should be (removeEmitted(annotations).toSeq)

    expected.foreach{ a =>
      info(s"$a was written")
      a should (exist)
      a.delete()
    }
  }

  it should "default to the output file name if one exists" in new Fixture {
    val annotations = Seq(
      TargetDirAnnotation("test_run_dir/WriteEmittedSpec"),
      OutputFileAnnotation("quux"),
      EmittedFirrtlCircuitAnnotation(EmittedFirrtlCircuit("qux", "", ".quxcircuit")) )
    val expected = new File("test_run_dir/WriteEmittedSpec/quux.quxcircuit")

    info("annotations are unmodified")
    phase.transform(annotations).toSeq should be (removeEmitted(annotations).toSeq)

    info(s"$expected was written")
    expected should (exist)
    expected.delete()
  }

  it should "write emitted modules" in new Fixture {
    val annotations = Seq(
      TargetDirAnnotation("test_run_dir/WriteEmittedSpec"),
      EmittedFirrtlModuleAnnotation(EmittedFirrtlModule("foo", "", ".foomodule")),
      EmittedFirrtlModuleAnnotation(EmittedFirrtlModule("bar", "", ".barmodule")),
      EmittedVerilogModuleAnnotation(EmittedVerilogModule("baz", "", ".bazmodule")) )
    val expected = Seq("foo.foomodule", "bar.barmodule", "baz.bazmodule")
      .map(a => new File(s"test_run_dir/WriteEmittedSpec/$a"))

    info("EmittedComponent annotations are deleted")
    phase.transform(annotations).toSeq should be (removeEmitted(annotations).toSeq)

    expected.foreach{ a =>
      info(s"$a was written")
      a should (exist)
      a.delete()
    }
  }

}
