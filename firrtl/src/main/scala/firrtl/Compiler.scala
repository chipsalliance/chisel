// SPDX-License-Identifier: Apache-2.0

package firrtl

import logger._
import java.io.Writer

import scala.collection.mutable
import scala.collection.immutable.VectorBuilder
import scala.util.Try
import scala.util.control.NonFatal
import firrtl.annotations._
import firrtl.ir.Circuit
import firrtl.Utils.throwInternalError
import firrtl.options.{Dependency, DependencyAPI, StageUtils, TransformLike}

/** Container of all annotations for a Firrtl compiler */
class AnnotationSeq private (underlying: Seq[Annotation]) {
  def toSeq: Seq[Annotation] = underlying
}
object AnnotationSeq {
  def apply(xs: Seq[Annotation]): AnnotationSeq = new AnnotationSeq(xs)
}

/** Current State of the Circuit
  *
  * @constructor Creates a CircuitState object
  * @param circuit The current state of the Firrtl AST
  * @param form The current form of the circuit
  * @param annotations The current collection of [[firrtl.annotations.Annotation Annotation]]
  * @param renames A map of [[firrtl.annotations.Named Named]] things that have been renamed.
  *   Generally only a return value from [[Transform]]s
  */
case class CircuitState(
  circuit:     Circuit,
  form:        CircuitForm,
  annotations: AnnotationSeq,
  renames:     Option[RenameMap]) {

  /** Helper for getting just an emitted circuit */
  def emittedCircuitOption: Option[EmittedCircuit] =
    emittedComponents.collectFirst { case x: EmittedCircuit => x }

  /** Helper for getting an [[EmittedCircuit]] when it is known to exist */
  def getEmittedCircuit: EmittedCircuit = emittedCircuitOption match {
    case Some(emittedCircuit) => emittedCircuit
    case None =>
      throw new FirrtlInternalException(
        s"No EmittedCircuit found! Did you delete any annotations?\n$deletedAnnotations"
      )
  }

  /** Helper function for extracting emitted components from annotations */
  def emittedComponents: Seq[EmittedComponent] =
    annotations.collect { case emitted: EmittedAnnotation[_] => emitted.value }
  def deletedAnnotations: Seq[Annotation] =
    annotations.collect { case anno: DeletedAnnotation => anno }

  /** Returns all annotations which are of a class in annoClasses
    * @param annoClasses
    * @return
    */
  def getAnnotationsOf(annoClasses: Class[_]*): AnnotationSeq = {
    annotations.collect { case a if annoClasses.contains(a.getClass) => a }
  }
}

object CircuitState {
  def apply(circuit: Circuit, form: CircuitForm): CircuitState = apply(circuit, form, Seq())
  def apply(circuit: Circuit, form: CircuitForm, annotations: AnnotationSeq): CircuitState =
    new CircuitState(circuit, form, annotations, None)
  def apply(circuit: Circuit, annotations: AnnotationSeq): CircuitState =
    new CircuitState(circuit, UnknownForm, annotations, None)
}

/** Current form of the Firrtl Circuit
  *
  * Form is a measure of addition restrictions on the legality of a Firrtl
  * circuit.  There is a notion of "highness" and "lowness" implemented in the
  * compiler by extending scala.math.Ordered. "Lower" forms add additional
  * restrictions compared to "higher" forms. This means that "higher" forms are
  * strictly supersets of the "lower" forms. Thus, that any transform that
  * operates on [[HighForm]] can also operate on [[MidForm]] or [[LowForm]]
  */
@deprecated(
  "Mix-in the DependencyAPIMigration trait into your Transform and specify its Dependency API dependencies. See: https://bit.ly/2Voppre",
  "FIRRTL 1.3"
)
sealed abstract class CircuitForm(private val value: Int) extends Ordered[CircuitForm] {
  // Note that value is used only to allow comparisons
  def compare(that: CircuitForm): Int = this.value - that.value

  /** Defines a suffix to use if this form is written to a file */
  def outputSuffix: String
}

/** Unknown Form
  *
  * Often passes may modify a circuit (e.g. InferTypes), but return
  * a circuit in the same form it was given.
  *
  * For this use case, use UnknownForm. It cannot be compared against other
  * forms.
  *
  * TODO(azidar): Replace with PreviousForm, which more explicitly encodes
  * this requirement.
  */
@deprecated(
  "Mix-in the DependencyAPIMigration trait into your Transform and specify its Dependency API dependencies. See: https://bit.ly/2Voppre",
  "FIRRTL 1.3"
)
final case object UnknownForm extends CircuitForm(-1) {
  override def compare(that: CircuitForm): Int = { sys.error("Illegal to compare UnknownForm"); 0 }

  val outputSuffix: String = ".unknown.fir"
}
