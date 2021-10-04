// SPDX-License-Identifier: Apache-2.0

package chisel3.stage.phases

import chisel3.Module
import chisel3.internal.ExceptionHelpers.ThrowableHelpers
import chisel3.internal.{Builder, DynamicContext}
import chisel3.stage.{ChiselCircuitAnnotation, ChiselGeneratorAnnotation, ChiselOptions, DesignAnnotation}
import firrtl.AnnotationSeq
import firrtl.options.Phase
import firrtl.options.Viewer.view

/** Elaborate all [[chisel3.stage.ChiselGeneratorAnnotation]]s into [[chisel3.stage.ChiselCircuitAnnotation]]s.
  */
class Elaborate extends Phase {

  override def prerequisites = Seq.empty
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf = Seq.empty
  override def invalidates(a: Phase) = false

  def transform(annotations: AnnotationSeq): AnnotationSeq = annotations.flatMap {
    case ChiselGeneratorAnnotation(gen) => try {
          val (circuit, dut) = Builder.build(Module(gen()), new DynamicContext(annotations))
          Seq(ChiselCircuitAnnotation(circuit), DesignAnnotation(dut))
    } catch {
      /* if any throwable comes back and we're in "stack trace trimming" mode, then print an error and trim the stack trace
       */
      case scala.util.control.NonFatal(a) =>
        if (!view[ChiselOptions](annotations).printFullStackTrace) {
          a.trimStackTraceToUserCode()
        }
        throw(a)
    }
    case a => Some(a)
  }

}
