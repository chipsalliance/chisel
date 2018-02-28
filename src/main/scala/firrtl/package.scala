
import firrtl.AnnotationSeq
import firrtl.annotations.Annotation

package object firrtl {
  implicit def seqToAnnoSeq(xs: Seq[Annotation]) = AnnotationSeq(xs)
  implicit def annoSeqToSeq(as: AnnotationSeq): Seq[Annotation] = as.underlying
}
