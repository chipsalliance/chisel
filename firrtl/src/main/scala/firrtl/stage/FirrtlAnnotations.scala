// SPDX-License-Identifier: Apache-2.0

package firrtl.stage

import firrtl._
import firrtl.ir.Circuit
import firrtl.annotations.{Annotation, NoTargetAnnotation}
import firrtl.options.{HasShellOptions, ShellOption, Unserializable}

/** Indicates that this is an [[firrtl.annotations.Annotation Annotation]] directly used in the construction of a
  * [[FirrtlOptions]] view.
  */
sealed trait FirrtlOption extends Unserializable { this: Annotation => }

/** An explicit output file the emitter will write to
  *   - set with `-o/--output-file`
  *  @param file output filename
  */
case class OutputFileAnnotation(file: String) extends NoTargetAnnotation with FirrtlOption

object OutputFileAnnotation extends HasShellOptions {

  val options = Seq(
    new ShellOption[String](
      longOption = "output-file",
      toAnnotationSeq = a => Seq(OutputFileAnnotation(a)),
      helpText = "The output FIRRTL file",
      shortOption = Some("o"),
      helpValueName = Some("<file>")
    )
  )

}

/** Sets the info mode style
  *  - set with `--info-mode`
  * @param mode info mode name
  */
case class InfoModeAnnotation(modeName: String = "use") extends NoTargetAnnotation with FirrtlOption {
  require(
    modeName match { case "use" | "ignore" | "gen" | "append" => true; case _ => false },
    s"Unknown info mode '$modeName'! (Did you misspell it?)"
  )

  /** Return the [[Parser.InfoMode]] equivalent for this [[firrtl.annotations.Annotation Annotation]]
    * @param infoSource the name of a file to use for "gen" or "append" info modes
    */
  def toInfoMode(infoSource: Option[String] = None): Parser.InfoMode = modeName match {
    case "use"    => Parser.UseInfo
    case "ignore" => Parser.IgnoreInfo
    case _ =>
      val a = infoSource.getOrElse("unknown source")
      modeName match {
        case "gen"    => Parser.GenInfo(a)
        case "append" => Parser.AppendInfo(a)
      }
  }
}

object InfoModeAnnotation extends HasShellOptions {

  val options = Seq(
    new ShellOption[String](
      longOption = "info-mode",
      toAnnotationSeq = a => Seq(InfoModeAnnotation(a)),
      helpText = s"Source file info handling mode (default: ${apply().modeName})",
      helpValueName = Some("<ignore|use|gen|append>")
    )
  )

}

/** Holds a FIRRTL [[firrtl.ir.Circuit Circuit]]
  * @param circuit a circuit
  */
case class FirrtlCircuitAnnotation(circuit: Circuit) extends NoTargetAnnotation with FirrtlOption {
  /* Caching the hashCode for a large circuit is necessary due to repeated queries, e.g., in
   * [[Compiler.propagateAnnotations]]. Not caching the hashCode will cause severe performance degredations for large
   * [[Annotations]].
   * @note Uses the hashCode of the name of the circuit. Creating a HashMap with different Circuits
   *       that nevertheless have the same name is extremely uncommon so collisions are not a concern.
   *       Include productPrefix so that this doesn't collide with other types that use a similar
   *       strategy and hash the same String.
   */
  override lazy val hashCode: Int = (this.productPrefix + circuit.main).hashCode

}

case object AllowUnrecognizedAnnotations extends NoTargetAnnotation with FirrtlOption with HasShellOptions {
  val options = Seq(
    new ShellOption[Unit](
      longOption = "allow-unrecognized-annotations",
      toAnnotationSeq = _ => Seq(this),
      helpText = "Allow annotation files to contain unrecognized annotations"
    )
  )
}
