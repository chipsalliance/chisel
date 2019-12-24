// See LICENSE for license details.

package chisel3.stage.phases

import java.io.{PrintWriter, StringWriter}

import chisel3.ChiselException
import chisel3.internal.{ErrorLog, Stash}
import chisel3.stage.{ChiselGeneratorAnnotation, ChiselOptions}
import firrtl.AnnotationSeq
import firrtl.options.Viewer.view
import firrtl.options.{OptionsException, Phase, PreservesAll}

/** Elaborate all [[chisel3.stage.ChiselGeneratorAnnotation]]s into [[chisel3.stage.ChiselCircuitAnnotation]]s.
  */
class FillStash extends Phase with PreservesAll[Phase] {

  def transform(annotations: AnnotationSeq): AnnotationSeq = annotations.flatMap {
    case stash: Stash if !stash.loaded && stash.backingStore.nonEmpty =>
      val dir = java.io.File(stash.backingStore.get)
    case a        => Some(a)
  }

}
