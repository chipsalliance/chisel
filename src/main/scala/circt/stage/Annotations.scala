// SPDX-License-Identifier: Apache-2.0

package circt.stage

import firrtl.AnnotationSeq
import firrtl.annotations.{Annotation, NoTargetAnnotation}
import firrtl.options.{CustomFileEmission, HasShellOptions, OptionsException, ShellOption, Unserializable}
import firrtl.options.Viewer.view
import firrtl.stage.FirrtlOptions

/** An option used to construct a [[circt.stage.CIRCTOptions CIRCTOptions]] */
sealed trait CIRCTOption extends Unserializable { this: Annotation => }

object PreserveAggregate extends HasShellOptions {
  sealed trait Type
  object OneDimVec extends Type
  object Vec extends Type
  object All extends Type

  override def options = Seq(
    new ShellOption[String](
      longOption = "preserve-aggregate",
      toAnnotationSeq = _ match {
        case "none"   => Seq.empty
        case "1d-vec" => Seq(PreserveAggregate(PreserveAggregate.OneDimVec))
        case "vec"    => Seq(PreserveAggregate(PreserveAggregate.Vec))
        case "all"    => Seq(PreserveAggregate(PreserveAggregate.All))
      },
      helpText = "Do not lower aggregate types to ground types"
    )
  )

}

/** Preserve passive aggregate types in CIRCT.
  */
case class PreserveAggregate(mode: PreserveAggregate.Type) extends NoTargetAnnotation with CIRCTOption

/** Object storing types associated with different CIRCT target languages, e.g., RTL or SystemVerilog */
object CIRCTTarget {

  /** The parent type of all CIRCT targets */
  sealed trait Type

  /** Specification FIRRTL */
  case object CHIRRTL extends Type

  /** The FIRRTL dialect */
  case object FIRRTL extends Type

  /** The HW dialect */
  case object HW extends Type

  /** The Verilog language */
  case object Verilog extends Type

  /** The SystemVerilog language */
  case object SystemVerilog extends Type

}

/** Annotation that tells [[circt.stage.phases.CIRCT CIRCT]] what target to compile to */
case class CIRCTTargetAnnotation(target: CIRCTTarget.Type) extends NoTargetAnnotation with CIRCTOption

object CIRCTTargetAnnotation extends HasShellOptions {

  override def options = Seq(
    new ShellOption[String](
      longOption = "target",
      toAnnotationSeq = _ match {
        case "chirrtl"       => Seq(CIRCTTargetAnnotation(CIRCTTarget.CHIRRTL))
        case "firrtl"        => Seq(CIRCTTargetAnnotation(CIRCTTarget.FIRRTL))
        case "hw"            => Seq(CIRCTTargetAnnotation(CIRCTTarget.HW))
        case "verilog"       => Seq(CIRCTTargetAnnotation(CIRCTTarget.Verilog))
        case "systemverilog" => Seq(CIRCTTargetAnnotation(CIRCTTarget.SystemVerilog))
        case a               => throw new OptionsException(s"Unknown target name '$a'! (Did you misspell it?)")
      },
      helpText = "The CIRCT",
      helpValueName = Some("{chirrtl|firrtl|hw|verilog|systemverilog}")
    )
  )

}

/** Annotation holding an emitted MLIR string
  *
  * @param filename the name of the file where this should be written
  * @param value a string of MLIR
  * @param suffix an optional suffix added to the filename when this is written to disk
  */
case class EmittedMLIR(
  filename: String,
  value:    String,
  suffix:   Option[String])
    extends NoTargetAnnotation
    with CustomFileEmission {

  override protected def baseFileName(annotations: AnnotationSeq): String = filename

  override def getBytes = value.getBytes

}

private[stage] case object FirtoolBinaryPath extends HasShellOptions {
  override def options = Seq(
    new ShellOption(
      longOption = "firtool-binary-path",
      toAnnotationSeq = (path: String) => Seq(FirtoolBinaryPath(path)),
      helpText = """Specifies the path to the "firtool" binary Chisel should use.""",
      helpValueName = Some("path")
    )
  )
}

/** Annotation that tells [[circt.stage.phases.CIRCT CIRCT]] what firtool executable to use */
case class FirtoolBinaryPath(option: String) extends NoTargetAnnotation with CIRCTOption

case class FirtoolOption(option: String) extends NoTargetAnnotation with CIRCTOption

/** Annotation that indicates that firtool should run using the
  * `--split-verilog` option.  This has two effects: (1) Verilog will be emitted
  * as one-file-per-module and (2) any other output file attributes created
  * along the way will have their operations written to other files.  Without
  * this option, output file attributes will have their operations emitted
  * inline in the single-file Verilog produced.
  */
private[circt] case object SplitVerilog extends NoTargetAnnotation with CIRCTOption with HasShellOptions {

  override def options = Seq(
    new ShellOption[Unit](
      longOption = "split-verilog",
      toAnnotationSeq = _ => Seq(this),
      helpText =
        """Indicates that "firtool" should emit one-file-per-module and write separate outputs to separate files""",
      helpValueName = None
    )
  )

}

/** Write the intermediate `.fir` file in [[circt.stage.ChiselStage]]
  */
private[circt] case object DumpFir extends NoTargetAnnotation with CIRCTOption with HasShellOptions {
  override def options = Seq(
    new ShellOption[Unit](
      longOption = "dump-fir",
      toAnnotationSeq = _ => Seq(this),
      helpText = "Write the intermediate .fir file",
      helpValueName = None
    )
  )

}
