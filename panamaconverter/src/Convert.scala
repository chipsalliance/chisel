// SPDX-License-Identifier: Apache-2.0

package chisel3.panamaconverter.stage

import chisel3.panamaconverter.PanamaCIRCTConverter
import chisel3.panamalib.option.FirtoolOptions
import chisel3.stage.ChiselCircuitAnnotation
import chisel3.stage.phases.Elaborate
import firrtl.AnnotationSeq
import firrtl.annotations.NoTargetAnnotation
import firrtl.options.{Dependency, Phase}

case class PanamaCIRCTConverterAnnotation(converter: PanamaCIRCTConverter) extends NoTargetAnnotation
case class FirtoolOptionsAnnotation(firtoolOptions: FirtoolOptions) extends NoTargetAnnotation

object Convert extends Phase {
  override def prerequisites = Seq(Dependency[Elaborate])
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf = Seq.empty
  override def invalidates(a: Phase) = false

  def transform(annotations: AnnotationSeq): AnnotationSeq =
    annotations.flatMap {
      case c @ ChiselCircuitAnnotation(circuit) =>
        Seq(
          c,
          PanamaCIRCTConverterAnnotation(
            PanamaCIRCTConverter.convert(
              circuit,
              annotations.collectFirst {
                case FirtoolOptionsAnnotation(firtoolOptions) => firtoolOptions
              },
              firrtl.annotations.JsonProtocol.serialize(circuit.firrtlAnnotations.toSeq)
            )
          )
        )
      case a => Seq(a)
    }
}
