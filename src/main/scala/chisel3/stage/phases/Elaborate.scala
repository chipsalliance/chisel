// See LICENSE for license details.

package chisel3.stage.phases


import chisel3.stage.ChiselGeneratorAnnotation
import firrtl.AnnotationSeq
import firrtl.options.{Phase, PreservesAll}

/** Elaborate all [[chisel3.stage.ChiselGeneratorAnnotation]]s into [[chisel3.stage.ChiselCircuitAnnotation]]s.
  */
class Elaborate extends Phase with PreservesAll[Phase] {

  def transform(annotations: AnnotationSeq): AnnotationSeq = annotations.flatMap {
    case a: ChiselGeneratorAnnotation => a.elaborate
    case a                            => Some(a)
  }

}
