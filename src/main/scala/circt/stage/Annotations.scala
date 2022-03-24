// SPDX-License-Identifier: Apache-2.0

package circt.stage

import firrtl.AnnotationSeq
import firrtl.annotations.{
  Annotation,
  NoTargetAnnotation
}
import firrtl.options.{
  CustomFileEmission,
  HasShellOptions,
  OptionsException,
  ShellOption,
  Unserializable
}
import firrtl.options.Viewer.view
import firrtl.stage.FirrtlOptions

/** An option consumed by [[circt.stage.CIRCTStage CIRCTStage]]*/
sealed trait CIRCTOption extends Unserializable { this: Annotation => }

/** Turns off type lowering in CIRCT. The option `-enable-lower-types` is passed by default to CIRCT and this annotation
  * turns that off.
  */
case object DisableLowerTypes extends NoTargetAnnotation with CIRCTOption with HasShellOptions {

  override def options = Seq(
    new ShellOption[Unit](
      longOption = "disable-lower-types",
      toAnnotationSeq = _ => Seq(DisableLowerTypes),
      helpText = "Do not lower aggregate types to ground types"
    )
  )

}

/** Object storing types associated with different CIRCT target languages, e.g., RTL or SystemVerilog */
object CIRCTTarget {

  /** The parent type of all CIRCT targets */
  sealed trait Type

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
        case "firrtl"        => Seq(CIRCTTargetAnnotation(CIRCTTarget.FIRRTL))
        case "hw"            => Seq(CIRCTTargetAnnotation(CIRCTTarget.HW))
        case "verilog"       => Seq(CIRCTTargetAnnotation(CIRCTTarget.Verilog))
        case "systemverilog" => Seq(CIRCTTargetAnnotation(CIRCTTarget.SystemVerilog))
        case a => throw new OptionsException(s"Unknown target name '$a'! (Did you misspell it?)")
      },
      helpText = "The CIRCT",
      helpValueName = Some("{firrtl|rtl|systemverilog}")
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
  value: String,
  suffix: Option[String]
) extends NoTargetAnnotation with CustomFileEmission {

  override protected def baseFileName(annotations: AnnotationSeq): String = filename

  override def getBytes = value.getBytes

}


object CIRCTHandover extends HasShellOptions {

  sealed trait Type

  case object CHIRRTL extends Type
  case object HighFIRRTL extends Type
  case object MiddleFIRRTL extends Type
  case object LowFIRRTL extends Type
  case object LowOptimizedFIRRTL extends Type

  override def options = Seq(
    new ShellOption[String](
      longOption = "handover",
      toAnnotationSeq = _ match {
        case "chirrtl" => Seq(CIRCTHandover(CHIRRTL))
        case "high" => Seq(CIRCTHandover(HighFIRRTL))
        case "middle" => Seq(CIRCTHandover(MiddleFIRRTL))
        case "low" => Seq(CIRCTHandover(LowFIRRTL))
        case "lowopt" => Seq(CIRCTHandover(LowOptimizedFIRRTL))
        case a => throw new OptionsException(s"Unknown handover point '$a'! (Did you misspell it?)")
      },
      helpText = "Switch to the CIRCT compiler at this point, using the Scala FIRRTL Compiler if needed",
      helpValueName = Some("{chirrtl|high|middle|low|lowopt}")
    )
  )
}

case class CIRCTHandover(handover: CIRCTHandover.Type) extends NoTargetAnnotation with CIRCTOption

case class FirtoolOption(option: String) extends NoTargetAnnotation with CIRCTOption
