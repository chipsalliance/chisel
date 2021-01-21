// SPDX-License-Identifier: Apache-2.0

package chisel3.stage.phases

import chisel3.internal.{Builder, DynamicContext}
import chisel3.stage.{ChiselCircuitAnnotation, ChiselGeneratorAnnotation, DesignAnnotation}
import chisel3.{ChiselException, Module}
import firrtl.AnnotationSeq
import firrtl.options.{OptionsException, Phase}

/** Elaborate all [[chisel3.stage.ChiselGeneratorAnnotation]]s into [[chisel3.stage.ChiselCircuitAnnotation]]s.
  */
class Elaborate extends Phase {

  override def prerequisites = Seq.empty
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf = Seq.empty
  override def invalidates(a: Phase) = false

  def transform(annotations: AnnotationSeq): AnnotationSeq = {
    try {
      annotations.flatMap{
        case ChiselGeneratorAnnotation(gen) =>
          val (circuit, dut) = Builder.build(Module(gen()), new DynamicContext(annotations))
          Seq(ChiselCircuitAnnotation(circuit), DesignAnnotation(dut))
        case a => Some(a)
      }
    } catch {
      case e @ (_: OptionsException | _: ChiselException) => throw e
      case e: Throwable =>
        throw new ChiselException(s"Exception thrown when elaborating ChiselGeneratorAnnotation", e)
    }
  }
}
