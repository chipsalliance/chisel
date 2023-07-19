// SPDX-License-Identifier: Apache-2.0

package chisel3.testers

import chisel3.RawModule
import chisel3.stage.ChiselGeneratorAnnotation
import chisel3.testers.TesterDriver.Backend
import chisel3.experimental.SourceInfo
import chisel3.internal.{Builder, Warning, WarningID}
import circt.stage.ChiselStage
import firrtl.AnnotationSeq
import firrtl.annotations.Annotation
import firrtl.ir.Circuit
import firrtl.stage.FirrtlCircuitAnnotation
import firrtl.util.BackendCompilationUtilities.createTestDirectory

object TestUtils {
  // Useful because TesterDriver.Backend is chisel3 package private
  def containsBackend(annos: AnnotationSeq): Boolean =
    annos.collectFirst { case b: Backend => b }.isDefined

  /** Helper for checking warnings, not really valid in normal Chisel */
  def warn(id: Int, msg: String)(implicit sourceInfo: SourceInfo): Unit = Builder.warning(Warning(WarningID(1), msg))

  // This used for backward support of test that rely on chisel3 chirrtl generation and access to the annotations
  // produced by it. New tests should not utilize this or getFirrtlAndAnnos
  def getChirrtlAndAnnotations(gen: => RawModule, annos: AnnotationSeq = Seq()): (Circuit, Seq[Annotation]) = {
    val dir = createTestDirectory(this.getClass.getSimpleName).toString
    val processedAnnos = (new ChiselStage).execute(
      Array("--target-dir", dir, "--target", "chirrtl"),
      ChiselGeneratorAnnotation(() => gen) +: annos
    )

    val circuit = processedAnnos.collectFirst {
      case FirrtlCircuitAnnotation(a) => a
    }.get
    (circuit, processedAnnos)
  }

}
