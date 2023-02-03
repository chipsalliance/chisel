// SPDX-License-Identifier: Apache-2.0

package chisel3.testers

import chisel3.RawModule
import chisel3.stage.{ChiselGeneratorAnnotation, ChiselPhase}
import chisel3.testers.TesterDriver.Backend
import circt.stage.{CIRCTTarget, CIRCTTargetAnnotation}
import firrtl.AnnotationSeq
import firrtl.annotations.Annotation
import firrtl.ir.Circuit
import firrtl.options.{Dependency, TargetDirAnnotation}
import firrtl.stage.FirrtlCircuitAnnotation
import firrtl.util.BackendCompilationUtilities.createTestDirectory
import org.scalatest.Assertions.fail

object TestUtils {
  // Useful because TesterDriver.Backend is chisel3 package private
  def containsBackend(annos: AnnotationSeq): Boolean =
    annos.collectFirst { case b: Backend => b }.isDefined

  // This used for backward support of test that rely on chisel3 chirrtl generation and access to the annotations
  // produced by it. New tests should not utilize this or getFirrtlAndAnnos
  def getChirrtlAndAnnotations(gen: => RawModule, annos: AnnotationSeq = Seq()): (Circuit, Seq[Annotation]) = {
    val dir = createTestDirectory(this.getClass.getSimpleName).toString
    val targetDir = TargetDirAnnotation(dir)
    val phase = new chisel3.stage.ChiselPhase {
      override val targets = Seq(
        Dependency[chisel3.stage.phases.Checks],
        Dependency[chisel3.stage.phases.Elaborate],
        Dependency[chisel3.stage.phases.AddImplicitOutputFile],
        Dependency[chisel3.stage.phases.AddImplicitOutputAnnotationFile],
        Dependency[chisel3.stage.phases.MaybeAspectPhase],
        Dependency[chisel3.stage.phases.Convert],
        Dependency[chisel3.stage.phases.MaybeInjectingPhase]
      )
    }

    val processedAnnos = phase
      .transform(Seq(ChiselGeneratorAnnotation(() => gen), targetDir) ++ annos)
    val circuit = processedAnnos.collectFirst {
      case FirrtlCircuitAnnotation(a) => a
    }.get
    (circuit, processedAnnos)
  }
}
