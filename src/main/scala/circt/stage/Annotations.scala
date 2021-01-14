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

sealed trait CIRCTOption extends Unserializable { this: Annotation => }

case object DisableLowerTypes extends NoTargetAnnotation with CIRCTOption with HasShellOptions {

  override def options = Seq(
    new ShellOption[Unit](
      longOption = "disable-lower-types",
      toAnnotationSeq = _ => Seq(DisableLowerTypes),
      helpText = "Do not lower aggregate types to ground types"
    )
  )

}

object CIRCTTarget {

  sealed trait Type

  case object FIRRTL extends Type
  case object RTL extends Type
  case object SystemVerilog extends Type

}

case class CIRCTTargetAnnotation(target: CIRCTTarget.Type) extends NoTargetAnnotation with CIRCTOption

object CIRCTTargetAnnotation extends HasShellOptions {

  override def options = Seq(
    new ShellOption[String](
      longOption = "target",
      toAnnotationSeq = _ match {
        case "firrtl"        => Seq(CIRCTTargetAnnotation(CIRCTTarget.FIRRTL))
        case "rtl"           => Seq(CIRCTTargetAnnotation(CIRCTTarget.RTL))
        case "systemverilog" => Seq(CIRCTTargetAnnotation(CIRCTTarget.SystemVerilog))
        case a => throw new OptionsException(s"Unknown target name '$a'! (Did you misspell it?)")
      },
      helpText = "The CIRCT",
      helpValueName = Some("{firrtl|rtl|systemverilog}")
    )
  )

}

case class EmittedMLIR(filename: String, value: String, suffix: Option[String]) extends NoTargetAnnotation with CustomFileEmission {

  override protected def baseFileName(annotations: AnnotationSeq): String = filename

  override def getBytes = value.getBytes

}
