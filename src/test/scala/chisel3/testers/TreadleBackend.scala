// See LICENSE for license details.

package chisel3.testers

import TesterDriver.createTestDirectory
import chisel3._
import chisel3.stage._
import firrtl.AnnotationSeq
import firrtl.annotations.NoTargetAnnotation
import firrtl.options.TargetDirAnnotation
import treadle.stage.TreadleTesterPhase
import treadle.executable.StopException
import treadle.{CallResetAtStartupAnnotation, TreadleTesterAnnotation, WriteVcdAnnotation}

import java.io.File

case object TreadleBackend extends TesterDriver.Backend {
  val MaxTreadleCycles = 10000L

  def execute(t:                    () => BasicTester,
              additionalVResources: Seq[String] = Seq(),
              annotations:          AnnotationSeq = Seq(),
              nameHint:             Option[String] = None): Boolean = {
    val generatorAnnotation = chisel3.stage.ChiselGeneratorAnnotation(t)

    // This provides an opportunity to translate from top level generic flags to backend specific annos
    var annotationSeq = annotations :+ WriteVcdAnnotation

    // This produces a chisel circuit annotation, a later pass will generate a firrtl circuit
    // Can't do both at once currently because generating the latter deletes the former
    annotationSeq = (new chisel3.stage.phases.Elaborate).transform(annotationSeq :+ generatorAnnotation)

    val circuit = annotationSeq.collect { case x: ChiselCircuitAnnotation => x }.head.circuit

    val targetName: File = createTestDirectory(circuit.name)

    if (!annotationSeq.exists(_.isInstanceOf[NoTargetAnnotation])) {
      annotationSeq = annotationSeq :+ TargetDirAnnotation(targetName.getPath)
    }
    if (!annotationSeq.exists { case CallResetAtStartupAnnotation => true ; case _ => false }) {
      annotationSeq = annotationSeq :+ CallResetAtStartupAnnotation
    }

    // This generates the firrtl circuit needed by the TreadleTesterPhase
    annotationSeq = (new ChiselStage).run(
      annotationSeq ++ Seq(NoRunFirrtlCompilerAnnotation)
    )

    // This generates a TreadleTesterAnnotation with a treadle tester instance
    annotationSeq = (new TreadleTesterPhase).transform(annotationSeq)

    val treadleTester = annotationSeq.collectFirst { case TreadleTesterAnnotation(t) => t }.getOrElse(
      throw new Exception(
        s"TreadleTesterPhase could not build a treadle tester from these annotations" +
          annotationSeq.mkString("Annotations:\n", "\n  ", "")
      )
    )

    try {
      var cycle = 0L
      while (cycle < MaxTreadleCycles) {
        cycle += 1
        treadleTester.step()
      }
      throw new ChiselException(s"Treadle backend exceeded MaxTreadleCycles ($MaxTreadleCycles)")
    } catch {
      case _: StopException =>
    }
    treadleTester.finish

    treadleTester.getStopResult match {
      case None    => true
      case Some(0) => true
      case _       => false
    }
  }
}
