// See LICENSE for license details.

import firrtl.annotations.Annotation

package object firrtl {
  // Force initialization of the Forms object - https://github.com/freechipsproject/firrtl/issues/1462
  private val _dummyForms = firrtl.stage.Forms

  implicit def seqToAnnoSeq(xs: Seq[Annotation]) = AnnotationSeq(xs)
  implicit def annoSeqToSeq(as: AnnotationSeq): Seq[Annotation] = as.underlying

  /* Options as annotations compatibility items */
  @deprecated("Use firrtl.stage.TargetDirAnnotation", "1.2")
  type TargetDirAnnotation = firrtl.options.TargetDirAnnotation

  @deprecated("Use firrtl.stage.TargetDirAnnotation", "1.2")
  val TargetDirAnnotation = firrtl.options.TargetDirAnnotation

  @deprecated("Use Flow instead of Gender. This trait will be removed in 1.3", "1.2")
  trait Gender

  @deprecated("Use SourceFlow instead of MALE. This case object will be removed in 1.3", "1.2")
  case object MALE extends Gender

  @deprecated("Use SinkFlow instead of FEMALE. This case object will be removed in 1.3", "1.2")
  case object FEMALE extends Gender

  @deprecated("Use DuplexFlow instead of BIGENDER. This case object will be removed in 1.3", "1.2")
  case object BIGENDER extends Gender

  @deprecated("Use UnknownFlow instead of UNKNOWNGENDER. This case object will be removed in 1.3", "1.2")
  case object UNKNOWNGENDER extends Gender

  @deprecated("Please migrate from 'Gender' to 'Flow'. This implicit conversion will be removed in 1.3", "1.2")
  implicit def genderToFlow(gender: Gender): Flow = gender match {
    case MALE          => SourceFlow
    case FEMALE        => SinkFlow
    case BIGENDER      => DuplexFlow
    case UNKNOWNGENDER => UnknownFlow
  }
}
