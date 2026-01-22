// SPDX-License-Identifier: Apache-2.0

package chisel3.stage.phases

import chisel3.Module
import chisel3.experimental.hierarchy.core.Definition
import chisel3.internal.ExceptionHelpers.ThrowableHelpers
import chisel3.internal.{Builder, BuilderContextCache, DynamicContext, ElaborationTrace}
import chisel3.internal.firrtl.ir
import chisel3.stage.{
  ChiselCircuitAnnotation,
  ChiselGeneratorAnnotation,
  ChiselOptions,
  ChiselOptionsView,
  DesignAnnotation,
  ThrowOnFirstErrorAnnotation
}
import firrtl.{annoSeqToSeq, seqToAnnoSeq, AnnotationSeq}
import firrtl.options.{Dependency, Phase}
import firrtl.options.Viewer.view
import logger.{LoggerOptions, LoggerOptionsView}

import scala.collection.mutable.ArrayBuffer
import scala.annotation.nowarn

/** Elaborate all [[chisel3.stage.ChiselGeneratorAnnotation]]s into [[chisel3.stage.ChiselCircuitAnnotation]]s.
  */
class Elaborate extends Phase {

  override def prerequisites: Seq[Dependency[Phase]] = Seq(
    Dependency[chisel3.stage.phases.Checks],
    Dependency(_root_.logger.phases.Checks)
  )
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf = Seq.empty
  override def invalidates(a: Phase) = false

  def transform(annotations: AnnotationSeq): AnnotationSeq = annotations.flatMap {
    case ChiselGeneratorAnnotation(gen) =>
      val chiselOptions = view[ChiselOptions](annotations)
      val loggerOptions = view[LoggerOptions](annotations)
      try {
        val elaborationTrace = new ElaborationTrace
        val context =
          new DynamicContext(
            annotations,
            chiselOptions.throwOnFirstError,
            chiselOptions.useLegacyWidth,
            chiselOptions.includeUtilMetadata,
            chiselOptions.useSRAMBlackbox,
            chiselOptions.warningFilters,
            chiselOptions.sourceRoots,
            None,
            loggerOptions,
            ArrayBuffer[Definition[_]](),
            BuilderContextCache.empty,
            chiselOptions.layerMap,
            chiselOptions.inlineTestIncluder,
            chiselOptions.suppressSourceInfo,
            false,
            elaborationTrace
          )
        val (elaboratedCircuit, dut) = {
          Builder.build(Module(gen()), context)
        }
        elaborationTrace.finish()

        // Extract the Chisel layers from a circuit via an in-order walk.
        def walkLayers(layer: ir.Layer, layers: Seq[chisel3.layer.Layer] = Nil): Seq[chisel3.layer.Layer] = {
          layer.children.foldLeft(layers :+ layer.chiselLayer) { case (acc, x) => walkLayers(x, acc) }
        }

        Seq(
          ChiselCircuitAnnotation(elaboratedCircuit),
          DesignAnnotation(dut, layers = elaboratedCircuit._circuit.layers.flatMap(walkLayers(_)))
        )
      } catch {
        /* if any throwable comes back and we're in "stack trace trimming" mode, then print an error and trim the stack trace
         */
        case scala.util.control.NonFatal(a) =>
          if (!chiselOptions.printFullStackTrace) {
            a.trimStackTraceToUserCode()
          }
          throw (a)
      }
    case a => Some(a)
  }

}
