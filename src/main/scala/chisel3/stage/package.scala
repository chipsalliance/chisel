// SPDX-License-Identifier: Apache-2.0

package chisel3

import firrtl._
import firrtl.options.OptionsView
import firrtl.{annoSeqToSeq, seqToAnnoSeq, AnnotationSeq}
import chisel3.internal.firrtl.ir.{Circuit => ChiselCircuit}
import chisel3.stage.CircuitSerializationAnnotation.FirrtlFileFormat

import scala.annotation.nowarn

package object stage {

  final val pleaseSwitchToCIRCT = deprecatedMFCMessage + " Please switch to circt.stage.ChiselStage."

  @nowarn("cat=deprecation&msg=Use elaboratedCircuit instead")
  implicit object ChiselOptionsView extends OptionsView[ChiselOptions] {

    def view(options: AnnotationSeq): ChiselOptions = options.collect { case a: ChiselOption => a }
      .foldLeft(new ChiselOptions()) { (c, x) =>
        x match {
          case PrintFullStackTraceAnnotation => c.copy(printFullStackTrace = true)
          case ThrowOnFirstErrorAnnotation   => c.copy(throwOnFirstError = true)
          case WarningsAsErrorsAnnotation =>
            c.copy(warningFilters = c.warningFilters :+ WarningsAsErrorsAnnotation.asFilter)
          case ChiselOutputFileAnnotation(f) => c.copy(outputFile = Some(f))
          case a: ChiselCircuitAnnotation =>
            c.copy(elaboratedCircuit = Some(a.elaboratedCircuit))
          case SourceRootAnnotation(s) => c.copy(sourceRoots = c.sourceRoots :+ s)
          case a: WarningConfigurationAnnotation     => c.copy(warningFilters = c.warningFilters ++ a.filters)
          case a: WarningConfigurationFileAnnotation => c.copy(warningFilters = c.warningFilters ++ a.filters)
          case UseLegacyWidthBehavior         => c.copy(useLegacyWidth = true)
          case RemapLayer(oldLayer, newLayer) => c.copy(layerMap = c.layerMap + ((oldLayer, newLayer)))
          case IncludeUtilMetadata            => c.copy(includeUtilMetadata = true)
          case UseSRAMBlackbox                => c.copy(useSRAMBlackbox = true)
          case IncludeInlineTestsForModuleAnnotation(glob) =>
            c.copy(inlineTestIncluder = c.inlineTestIncluder.includeModule(glob))
          case IncludeInlineTestsWithNameAnnotation(glob) =>
            c.copy(inlineTestIncluder = c.inlineTestIncluder.includeTest(glob))
          case SuppressSourceLocatorsAnnotation => c.copy(suppressSourceLocators = true)
        }
      }

  }
}
