package chisel3.util.experimental

import chisel3.internal.Builder
import firrtl.{annoSeqToSeq, seqToAnnoSeq, AnnotationSeq}

object getAnnotations {

  /** Returns the global Annotations */
  def apply(): AnnotationSeq = Builder.annotationSeq
}
