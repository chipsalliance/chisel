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

// Internal utilities to keep code DRY, not a clean interface
private[firrtl] object Transform {

  def remapAnnotations(after: CircuitState, logger: Logger): CircuitState = {
    val remappedAnnotations = propagateAnnotations(after.annotations, after.renames)

    logger.trace(s"Annotations:")
    logger.trace(JsonProtocol.serializeRecover(remappedAnnotations))

    logger.trace(s"Circuit:\n${after.circuit.serialize}")

    CircuitState(after.circuit, after.form, remappedAnnotations, None)
  }

  // This function is *very* mutable but it is fairly performance critical
  def propagateAnnotations(
    resAnno:   AnnotationSeq,
    renameOpt: Option[RenameMap]
  ): AnnotationSeq = {
    // We dedup/distinct the resulting annotations when renaming occurs
    val seen = new mutable.HashSet[Annotation]
    val result = new VectorBuilder[Annotation]

    val hasRenames = renameOpt.isDefined
    val renames = renameOpt.getOrElse(null) // Null is bad but saving the allocation is worth it

    val it = resAnno.toSeq.iterator
    while (it.hasNext) {
      val anno = it.next()
      if (hasRenames) {
        val renamed = anno.update(renames)
        for (annox <- renamed) {
          if (!seen(annox)) {
            seen += annox
            result += annox
          }
        }
      } else {
        result += anno
      }
    }
    result.result()
  }
}

/** The basic unit of operating on a Firrtl AST */
trait Transform extends TransformLike[CircuitState] with DependencyAPI[Transform] {

  /** A convenience function useful for debugging and error messages */
  def name: String = this.getClass.getName

  /** The [[firrtl.CircuitForm]] that this transform requires to operate on */
  @deprecated("Use Dependency API methods for equivalent functionality. See: https://bit.ly/2Voppre", "FIRRTL 1.3")
  def inputForm: CircuitForm

  /** The [[firrtl.CircuitForm]] that this transform outputs */
  @deprecated("Use Dependency API methods for equivalent functionality. See: https://bit.ly/2Voppre", "FIRRTL 1.3")
  def outputForm: CircuitForm

  /** Perform the transform, encode renaming with RenameMap, and can
    *   delete annotations
    * Called by [[runTransform]].
    *
    * @param state Input Firrtl AST
    * @return A transformed Firrtl AST
    */
  protected def execute(state: CircuitState): CircuitState

  def transform(state: CircuitState): CircuitState = execute(state)

  override def prerequisites: Seq[Dependency[Transform]] = Seq.empty

  override def optionalPrerequisites: Seq[Dependency[Transform]] = Seq.empty

  override def optionalPrerequisiteOf: Seq[Dependency[Transform]] = Seq.empty

  override def invalidates(a: Transform): Boolean = false

  /** Executes before any transform's execute method
    * @param state
    * @return
    */
  private[firrtl] def prepare(state: CircuitState): CircuitState = state

  /** Perform the transform and update annotations.
    *
    * @param state Input Firrtl AST
    * @return A transformed Firrtl AST
    */
  final def runTransform(state: CircuitState): CircuitState = {
    val result = execute(prepare(state))
    Transform.remapAnnotations(result, logger)
  }

}

trait SeqTransformBased {
  def transforms: Seq[Transform]
  protected def runTransforms(state: CircuitState): CircuitState =
    transforms.foldLeft(state) { (in, xform) => xform.runTransform(in) }
}

/** For transformations that are simply a sequence of transforms */
abstract class SeqTransform extends Transform with SeqTransformBased {
  def execute(state: CircuitState): CircuitState = {
    /*
    require(state.form <= inputForm,
      s"[$name]: Input form must be lower or equal to $inputForm. Got ${state.form}")
     */
    val ret = runTransforms(state)
    CircuitState(ret.circuit, outputForm, ret.annotations, ret.renames)
  }
}

/** Defines old API for Emission. Deprecated */
trait Emitter extends Transform {

  override def invalidates(a: Transform) = false

  @deprecated("Use emission annotations instead", "FIRRTL 1.0")
  def emit(state: CircuitState, writer: Writer): Unit

  /** An output suffix to use if the output of this [[Emitter]] was written to a file */
  def outputSuffix: String
}

@deprecated("Migrate to firrtl.stage.transforms.Compiler. This will be removed in 1.4.", "FIRRTL 1.3")
trait Compiler extends Transform with DependencyAPIMigration {
  def emitter: Emitter

  /** The sequence of transforms this compiler will execute
    * @note The inputForm of a given transform must be higher than or equal to the ouputForm of the
    *       preceding transform. See [[CircuitForm]]
    */
  def transforms: Seq[Transform]

  final override def execute(state: CircuitState): CircuitState =
    new stage.transforms.Compiler(
      targets = (transforms :+ emitter).map(Dependency.fromTransform),
      currentState = prerequisites,
      knownObjects = (transforms :+ emitter).toSet
    ).execute(state)

  require(
    transforms.size >= 1,
    s"Compiler transforms for '${this.getClass.getName}' must have at least ONE Transform! " +
      "Use IdentityTransform if you need an identity/no-op transform."
  )

  /** Perform compilation
    *
    * @param state The Firrtl AST to compile
    * @param writer The java.io.Writer where the output of compilation will be emitted
    * @param customTransforms Any custom [[Transform]]s that will be inserted
    *   into the compilation process by [[CompilerUtils.mergeTransforms]]
    */
  @deprecated(
    "Migrate to '(new FirrtlStage).execute(args: Array[String], annotations: AnnotationSeq)'." +
      "This will be removed in 1.4.",
    "FIRRTL 1.0"
  )
  def compile(state: CircuitState, writer: Writer, customTransforms: Seq[Transform] = Seq.empty): CircuitState = {
    val finalState = compileAndEmit(state, customTransforms)
    writer.write(finalState.getEmittedCircuit.value)
    finalState
  }

  /** Perform compilation and emit the whole Circuit
    *
    * This is intended as a convenience method wrapping up Annotation creation for the common case.
    * It creates a [[EmitCircuitAnnotation]] that will be consumed by this Transform's emitter. The
    * [[EmittedCircuit]] can be extracted from the returned [[CircuitState]] via
    * [[CircuitState.emittedCircuitOption]]
    *
    * @param state The Firrtl AST to compile
    * @param customTransforms Any custom [[Transform]]s that will be inserted
    *   into the compilation process by [[CompilerUtils.mergeTransforms]]
    * @return result of compilation with emitted circuit annotated
    */
  @deprecated(
    "Migrate to '(new FirrtlStage).execute(args: Array[String], annotations: AnnotationSeq)'." +
      "This will be removed in 1.4.",
    "FIRRTL 1.3.3"
  )
  def compileAndEmit(state: CircuitState, customTransforms: Seq[Transform] = Seq.empty): CircuitState = {
    val emitAnno = EmitCircuitAnnotation(emitter.getClass)
    compile(state.copy(annotations = emitAnno +: state.annotations), emitter +: customTransforms)
  }

  /** Perform compilation
    *
    * Emission will only be performed if [[EmitAnnotation]]s are present
    *
    * @param state The Firrtl AST to compile
    * @param customTransforms Any custom [[Transform]]s that will be inserted into the compilation
    *   process by [[CompilerUtils.mergeTransforms]]
    * @return result of compilation
    */
  @deprecated(
    "Migrate to '(new FirrtlStage).execute(args: Array[String], annotations: AnnotationSeq)'." +
      "This will be removed in 1.4.",
    "FIRRTL 1.3.3"
  )
  def compile(state: CircuitState, customTransforms: Seq[Transform]): CircuitState = {
    val transformManager = new stage.transforms.Compiler(
      targets = (emitter +: customTransforms ++: transforms).map(Dependency.fromTransform),
      currentState = prerequisites,
      knownObjects = (transforms :+ emitter).toSet
    )
    transformManager.transform(state)
  }

}
