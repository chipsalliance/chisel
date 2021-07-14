// SPDX-License-Identifier: Apache-2.0

package firrtl

import logger._
import java.io.Writer

import scala.collection.mutable
import scala.util.Try
import scala.util.control.NonFatal
import firrtl.annotations._
import firrtl.ir.Circuit
import firrtl.Utils.throwInternalError
import firrtl.annotations.transforms.{EliminateTargetPaths, ResolvePaths}
import firrtl.options.{Dependency, DependencyAPI, StageUtils, TransformLike}
import firrtl.stage.Forms
import firrtl.transforms.DedupAnnotationsTransform

/** Container of all annotations for a Firrtl compiler */
class AnnotationSeq private (private[firrtl] val underlying: List[Annotation]) {
  def toSeq: Seq[Annotation] = underlying.toSeq
}
object AnnotationSeq {
  def apply(xs: Seq[Annotation]): AnnotationSeq = new AnnotationSeq(xs.toList)
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

  /** Returns a new CircuitState with all targets being resolved.
    * Paths through instances are replaced with a uniquified final target
    * Includes modifying the circuit and annotations
    * @param targets
    * @return
    */
  def resolvePaths(targets: Seq[CompleteTarget]): CircuitState = targets match {
    case Nil => this
    case _ =>
      val newCS = new EliminateTargetPaths().runTransform(this.copy(annotations = ResolvePaths(targets) +: annotations))
      newCS.copy(form = form)
  }

  /** Returns a new CircuitState with the targets of every annotation of a type in annoClasses
    * @param annoClasses
    * @return
    */
  def resolvePathsOf(annoClasses: Class[_]*): CircuitState = {
    val targets = getAnnotationsOf(annoClasses: _*).flatMap(_.getTargets)
    if (targets.nonEmpty) resolvePaths(targets.flatMap { _.getComplete }) else this
  }

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
private[firrtl] object CircuitForm {
  // Private internal utils to reduce number of deprecation warnings
  val ChirrtlForm = firrtl.ChirrtlForm
  val HighForm = firrtl.HighForm
  val MidForm = firrtl.MidForm
  val LowForm = firrtl.LowForm
  val UnknownForm = firrtl.UnknownForm
}

// These magic numbers give an ordering to CircuitForm
/** Chirrtl Form
  *
  * The form of the circuit emitted by Chisel. Not a true Firrtl form.
  * Includes cmem, smem, and mport IR nodes which enable declaring memories
  * separately form their ports. A "Higher" form than [[HighForm]]
  *
  * See [[CDefMemory]] and [[CDefMPort]]
  */
@deprecated(
  "Mix-in the DependencyAPIMigration trait into your Transform and specify its Dependency API dependencies. See: https://bit.ly/2Voppre",
  "FIRRTL 1.3"
)
final case object ChirrtlForm extends CircuitForm(value = 3) {
  val outputSuffix: String = ".fir"
}

/** High Form
  *
  * As detailed in the Firrtl specification
  * [[https://github.com/ucb-bar/firrtl/blob/master/spec/spec.pdf]]
  *
  * Also see [[firrtl.ir]]
  */
@deprecated(
  "Mix-in the DependencyAPIMigration trait into your Transform and specify its Dependency API dependencies. See: https://bit.ly/2Voppre",
  "FIRRTL 1.3"
)
final case object HighForm extends CircuitForm(2) {
  val outputSuffix: String = ".hi.fir"
}

/** Middle Form
  *
  * A "lower" form than [[HighForm]] with the following restrictions:
  *  - All widths must be explicit
  *  - All whens must be removed
  *  - There can only be a single connection to any element
  */
@deprecated(
  "Mix-in the DependencyAPIMigration trait into your Transform and specify its Dependency API dependencies. See: https://bit.ly/2Voppre",
  "FIRRTL 1.3"
)
final case object MidForm extends CircuitForm(1) {
  val outputSuffix: String = ".mid.fir"
}

/** Low Form
  *
  * The "lowest" form. In addition to the restrictions in [[MidForm]]:
  *  - All aggregate types (vector/bundle) must have been removed
  *  - All implicit truncations must be made explicit
  */
@deprecated(
  "Mix-in the DependencyAPIMigration trait into your Transform and specify its Dependency API dependencies. See: https://bit.ly/2Voppre",
  "FIRRTL 1.3"
)
final case object LowForm extends CircuitForm(0) {
  val outputSuffix: String = ".lo.fir"
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

  def remapAnnotations(name: String, before: CircuitState, after: CircuitState, logger: Logger): CircuitState = {
    val remappedAnnotations = propagateAnnotations(name, logger, before.annotations, after.annotations, after.renames)

    logger.trace(s"Annotations:")
    logger.trace(JsonProtocol.serializeRecover(remappedAnnotations))

    logger.trace(s"Circuit:\n${after.circuit.serialize}")

    CircuitState(after.circuit, after.form, remappedAnnotations, None)
  }

  /** Propagate annotations and update their names.
    *
    * @param inAnno input AnnotationSeq
    * @param resAnno result AnnotationSeq
    * @param renameOpt result RenameMap
    * @return the updated annotations
    */
  def propagateAnnotations(
    name:      String,
    logger:    Logger,
    inAnno:    AnnotationSeq,
    resAnno:   AnnotationSeq,
    renameOpt: Option[RenameMap]
  ): AnnotationSeq = {
    val newAnnotations = {
      val inSet = mutable.LinkedHashSet() ++ inAnno
      val resSet = mutable.LinkedHashSet() ++ resAnno
      val deleted = (inSet -- resSet).map {
        case DeletedAnnotation(xFormName, delAnno) => DeletedAnnotation(s"$xFormName+$name", delAnno)
        case anno                                  => DeletedAnnotation(name, anno)
      }
      val created = resSet -- inSet
      val unchanged = resSet & inSet
      (deleted ++ created ++ unchanged)
    }

    // For each annotation, rename all annotations.
    val renames = renameOpt.getOrElse(RenameMap())
    val remapped2original = mutable.LinkedHashMap[Annotation, mutable.LinkedHashSet[Annotation]]()
    val keysOfNote = mutable.LinkedHashSet[Annotation]()
    val finalAnnotations = newAnnotations.flatMap { anno =>
      val remappedAnnos = anno.update(renames)
      remappedAnnos.foreach { remapped =>
        val set = remapped2original.getOrElseUpdate(remapped, mutable.LinkedHashSet.empty[Annotation])
        set += anno
        if (set.size > 1) keysOfNote += remapped
      }
      remappedAnnos
    }.toSeq
    keysOfNote.foreach { key =>
      logger.debug(s"""The following original annotations are renamed to the same new annotation.""")
      logger.debug(s"""Original Annotations:\n  ${remapped2original(key).mkString("\n  ")}""")
      logger.debug(s"""New Annotation:\n  $key""")
    }
    finalAnnotations
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

  import firrtl.CircuitForm.{ChirrtlForm => C, HighForm => H, MidForm => M, LowForm => L, UnknownForm => U}

  override def prerequisites: Seq[Dependency[Transform]] = inputForm match {
    case C => Nil
    case H => Forms.Deduped
    case M => Forms.MidForm
    case L => Forms.LowForm
    case U => Nil
  }

  override def optionalPrerequisites: Seq[Dependency[Transform]] = inputForm match {
    case L => Forms.LowFormOptimized ++ Forms.AssertsRemoved
    case _ => Seq.empty
  }

  private lazy val fullCompilerSet = new mutable.LinkedHashSet[Dependency[Transform]] ++ Forms.VerilogOptimized

  override def optionalPrerequisiteOf: Seq[Dependency[Transform]] = {
    val lowEmitters = Dependency[LowFirrtlEmitter] :: Dependency[VerilogEmitter] :: Dependency[MinimumVerilogEmitter] ::
      Dependency[SystemVerilogEmitter] :: Nil

    val emitters = inputForm match {
      case C =>
        Dependency[ChirrtlEmitter] :: Dependency[HighFirrtlEmitter] :: Dependency[MiddleFirrtlEmitter] :: lowEmitters
      case H => Dependency[HighFirrtlEmitter] :: Dependency[MiddleFirrtlEmitter] :: lowEmitters
      case M => Dependency[MiddleFirrtlEmitter] :: lowEmitters
      case L => lowEmitters
      case U => Nil
    }

    val selfDep = Dependency.fromTransform(this)

    inputForm match {
      case C => (fullCompilerSet ++ emitters - selfDep).toSeq
      case H => (fullCompilerSet -- Forms.Deduped ++ emitters - selfDep).toSeq
      case M => (fullCompilerSet -- Forms.MidForm ++ emitters - selfDep).toSeq
      case L => (fullCompilerSet -- Forms.LowFormOptimized ++ emitters - selfDep).toSeq
      case U => Nil
    }
  }

  private lazy val highOutputInvalidates = fullCompilerSet -- Forms.MinimalHighForm
  private lazy val midOutputInvalidates = fullCompilerSet -- Forms.MidForm

  override def invalidates(a: Transform): Boolean = {
    (inputForm, outputForm) match {
      case (U, _) | (_, U)  => true // invalidate everything
      case (i, o) if i >= o => false // invalidate nothing
      case (_, C)           => true // invalidate everything
      case (_, H)           => highOutputInvalidates(Dependency.fromTransform(a))
      case (_, M)           => midOutputInvalidates(Dependency.fromTransform(a))
      case (_, L)           => false // invalidate nothing
    }
  }

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
    Transform.remapAnnotations(name, state, result, logger)
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

/** Extend for transforms that require resolved targets in their annotations
  * Ensures all targets in annotations of a class in annotationClasses are resolved before the execute method
  */
trait ResolvedAnnotationPaths {
  this: Transform =>

  val annotationClasses: Traversable[Class[_]]

  override def prepare(state: CircuitState): CircuitState = {
    state.resolvePathsOf(annotationClasses.toSeq: _*)
  }

  // Any transform with this trait invalidates DedupAnnotationsTransform
  override def invalidates(a: Transform) = a.isInstanceOf[DedupAnnotationsTransform]
}

/** Defines old API for Emission. Deprecated */
trait Emitter extends Transform {

  override def invalidates(a: Transform) = false

  @deprecated("Use emission annotations instead", "FIRRTL 1.0")
  def emit(state: CircuitState, writer: Writer): Unit

  /** An output suffix to use if the output of this [[Emitter]] was written to a file */
  def outputSuffix: String
}

@deprecated("This will be removed in 1.4", "FIRRTL 1.3")
object CompilerUtils extends LazyLogging {

  /** Generates a sequence of [[Transform]]s to lower a Firrtl circuit
    *
    * @param inputForm [[CircuitForm]] to lower from
    * @param outputForm [[CircuitForm]] to lower to
    * @return Sequence of transforms that will lower if outputForm is lower than inputForm
    */
  @deprecated(
    "Use a TransformManager requesting which transforms you want to run. This will be removed in 1.4.",
    "FIRRTL 1.3"
  )
  def getLoweringTransforms(inputForm: CircuitForm, outputForm: CircuitForm): Seq[Transform] = {
    // If outputForm is equal-to or higher than inputForm, nothing to lower
    if (outputForm >= inputForm) {
      Seq.empty
    } else {
      inputForm match {
        case ChirrtlForm =>
          Seq(new ChirrtlToHighFirrtl) ++ getLoweringTransforms(HighForm, outputForm)
        case HighForm =>
          Seq(
            new IRToWorkingIR,
            new ResolveAndCheck,
            new firrtl.transforms.DedupModules,
            new HighFirrtlToMiddleFirrtl
          ) ++
            getLoweringTransforms(MidForm, outputForm)
        case MidForm     => Seq(new MiddleFirrtlToLowFirrtl) ++ getLoweringTransforms(LowForm, outputForm)
        case LowForm     => throwInternalError("getLoweringTransforms - LowForm") // should be caught by if above
        case UnknownForm => throwInternalError("getLoweringTransforms - UnknownForm") // should be caught by if above
      }
    }
  }

  /** Merge a Seq of lowering transforms with custom transforms
    *
    * Custom  Transforms are  inserted  based on  their [[Transform.inputForm]]  and  [[Transform.outputForm]] with  any
    * [[Emitter]]s being  scheduled last. Custom transforms  are inserted in  order at the  last location in the  Seq of
    * transforms where previous.outputForm == customTransform.inputForm. If a customTransform outputs a higher form than
    * input, [[getLoweringTransforms]] is used to relower the circuit.
    *
    * @example
    *   {{{
    *     // Let Transforms be represented by CircuitForm => CircuitForm
    *     val A = HighForm => MidForm
    *     val B = MidForm => LowForm
    *     val lowering = List(A, B) // Assume these transforms are used by getLoweringTransforms
    *     // Some custom transforms
    *     val C = LowForm => LowForm
    *     val D = MidForm => MidForm
    *     val E = LowForm => HighForm
    *     // All of the following comparisons are true
    *     mergeTransforms(lowering, List(C)) == List(A, B, C)
    *     mergeTransforms(lowering, List(D)) == List(A, D, B)
    *     mergeTransforms(lowering, List(E)) == List(A, B, E, A, B)
    *     mergeTransforms(lowering, List(C, E)) == List(A, B, C, E, A, B)
    *     mergeTransforms(lowering, List(E, C)) == List(A, B, E, A, B, C)
    *     // Notice that in the following, custom transform order is NOT preserved (see note)
    *     mergeTransforms(lowering, List(C, D)) == List(A, D, B, C)
    *   }}}
    *
    * @note Order will be preserved for custom transforms so long as the
    * inputForm of a latter transforms is equal to or lower than the outputForm
    * of the previous transform.
    */
  @deprecated(
    "Use a TransformManager requesting which transforms you want to run. This will be removed in 1.4.",
    "FIRRTL 1.3"
  )
  def mergeTransforms(lowering: Seq[Transform], custom: Seq[Transform]): Seq[Transform] = {
    custom.sortWith {
      case (a, b) =>
        (a, b) match {
          case (_: Emitter, _: Emitter) => false
          case (_, _: Emitter) => true
          case _ => false
        }
    }
      .foldLeft(lowering) {
        case (transforms, xform) =>
          val index = transforms.lastIndexWhere(_.outputForm == xform.inputForm)
          assert(
            index >= 0 || xform.inputForm == ChirrtlForm, // If ChirrtlForm just put at front
            s"No transform in $lowering has outputForm ${xform.inputForm} as required by $xform"
          )
          val (front, back) = transforms.splitAt(index + 1) // +1 because we want to be AFTER index
          front ++ List(xform) ++ getLoweringTransforms(xform.outputForm, xform.inputForm) ++ back
      }
  }

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
