// SPDX-License-Identifier: Apache-2.0

package firrtl

import firrtl.annotations.NoTargetAnnotation
import firrtl.options.Viewer.view
import firrtl.options.CustomFileEmission
import firrtl.stage.FirrtlOptions
import firrtl.stage.FirrtlOptionsView

// ***** Annotations for results of emission *****
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
sealed abstract class EmittedComponent {
  def name: String

  def value: String

  def outputSuffix: String
}

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
sealed abstract class EmittedCircuit extends EmittedComponent

/** Traits for Annotations containing emitted components */
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
trait EmittedAnnotation[T <: EmittedComponent] extends NoTargetAnnotation with CustomFileEmission {
  val value: T

  override protected def baseFileName(annotations: AnnotationSeq): String = {
    view[FirrtlOptions](annotations).outputFileName.getOrElse(value.name)
  }

  override protected val suffix: Option[String] = Some(value.outputSuffix)
}

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
sealed trait EmittedCircuitAnnotation[T <: EmittedCircuit] extends EmittedAnnotation[T] {
  override def getBytes = value.value.getBytes
}

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class EmittedFirrtlCircuitAnnotation(value: EmittedFirrtlCircuit)
    extends EmittedCircuitAnnotation[EmittedFirrtlCircuit]

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
final case class EmittedFirrtlCircuit(name: String, value: String, outputSuffix: String) extends EmittedCircuit

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
final case class EmittedBtor2Circuit(name: String, value: String, outputSuffix: String) extends EmittedCircuit

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class EmittedBtor2CircuitAnnotation(value: EmittedBtor2Circuit)
    extends EmittedCircuitAnnotation[EmittedBtor2Circuit]

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
final case class EmittedVerilogCircuit(name: String, value: String, outputSuffix: String) extends EmittedCircuit
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class EmittedVerilogCircuitAnnotation(value: EmittedVerilogCircuit)
    extends EmittedCircuitAnnotation[EmittedVerilogCircuit]
