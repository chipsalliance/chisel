// SPDX-License-Identifier: Apache-2.0

package firrtl.stage.phases

import firrtl.stage._

import firrtl.AnnotationSeq
import firrtl.annotations.Annotation
import firrtl.options.{Dependency, OptionsException, Phase}

/** [[firrtl.options.Phase Phase]] that strictly validates an [[AnnotationSeq]]. The checks applied are intended to be
  * extremeley strict. Nothing is inferred or assumed to take a default value (for default value resolution see
  * [[AddDefaults]]).
  *
  * The intent of this approach is that after running this [[firrtl.options.Phase Phase]], a user can be absolutely
  * certain that other [[firrtl.options.Phase Phase]]s or views will succeed.
  */
class Checks extends Phase {

  override val prerequisites = Seq(Dependency[AddDefaults])

  override val optionalPrerequisiteOf = Seq.empty

  override def invalidates(a: Phase) = false

  /** Determine if annotations are sane
    *
    * @param annos a sequence of [[firrtl.annotations.Annotation Annotation]]
    * @return true if all checks pass
    * @throws firrtl.options.OptionsException if any checks fail
    */
  def transform(annos: AnnotationSeq): AnnotationSeq = {
    val outF, im, inC = collection.mutable.ListBuffer[Annotation]()
    annos.foreach(_ match {
      case a: OutputFileAnnotation    => a +=: outF
      case a: InfoModeAnnotation      => a +=: im
      case a: FirrtlCircuitAnnotation => a +=: inC
      case _ =>
    })

    /* At this point, only a FIRRTL Circuit should exist */
    if (inC.isEmpty) {
      throw new OptionsException(
        s"""|Unable to determine FIRRTL source to read. None of the following were found:
            |    - FIRRTL circuit:                        FirrtlCircuitAnnotation""".stripMargin
      )
    }

    /* Only one FIRRTL input can exist */
    if (inC.size > 1) {
      throw new OptionsException(
        s"""|Multiply defined input FIRRTL sources. More than one of the following was found:
            |    - FIRRTL circuit (${inC.size} times):                        FirrtlCircuitAnnotation""".stripMargin
      )
    }

    /* Only one output file can be specified */
    if (outF.size > 1) {
      val x = outF.map { case OutputFileAnnotation(x) => x }
      throw new OptionsException(
        s"""|No more than one output file can be specified, but found '${x.mkString(", ")}' specified via:
            |    - option or annotation: -o, --output-file, OutputFileAnnotation""".stripMargin
      )
    }

    /* One mandatory info mode must be specified */
    if (im.size != 1) {
      val x = im.map { case InfoModeAnnotation(x) => x }
      val (msg, suggest) = if (im.size == 0) { ("none found", "forget one of") }
      else { (s"""found '${x.mkString(", ")}'""", "use multiple of") }
      throw new OptionsException(s"""|Exactly one info mode must be specified, but $msg. Did you $suggest the following?
                                     |    - an option or annotation: --info-mode, InfoModeAnnotation""".stripMargin)
    }

    annos
  }

}
