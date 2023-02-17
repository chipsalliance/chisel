// SPDX-License-Identifier: Apache-2.0

package firrtl

import firrtl.annotations.NoTargetAnnotation
import firrtl.options.Viewer.view
import firrtl.options.CustomFileEmission
import firrtl.stage.FirrtlOptions

// ***** Annotations for results of emission *****
sealed abstract class EmittedComponent {
  def name: String

  def value: String

  def outputSuffix: String
}

sealed abstract class EmittedCircuit extends EmittedComponent

/** Traits for Annotations containing emitted components */
trait EmittedAnnotation[T <: EmittedComponent] extends NoTargetAnnotation with CustomFileEmission {
  val value: T

  override protected def baseFileName(annotations: AnnotationSeq): String = {
    view[FirrtlOptions](annotations).outputFileName.getOrElse(value.name)
  }

  override protected val suffix: Option[String] = Some(value.outputSuffix)
}

sealed trait EmittedCircuitAnnotation[T <: EmittedCircuit] extends EmittedAnnotation[T] {
  override def getBytes = value.value.getBytes
}

case class EmittedFirrtlCircuitAnnotation(value: EmittedFirrtlCircuit)
    extends EmittedCircuitAnnotation[EmittedFirrtlCircuit]

final case class EmittedFirrtlCircuit(name: String, value: String, outputSuffix: String) extends EmittedCircuit

final case class EmittedVerilogCircuit(name: String, value: String, outputSuffix: String) extends EmittedCircuit
case class EmittedVerilogCircuitAnnotation(value: EmittedVerilogCircuit)
    extends EmittedCircuitAnnotation[EmittedVerilogCircuit]
