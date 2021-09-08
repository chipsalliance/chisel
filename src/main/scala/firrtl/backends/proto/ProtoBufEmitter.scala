// SPDX-License-Identifier: Apache-2.0
package firrtl.backends.proto

import firrtl._
import firrtl.ir._
import firrtl.annotations.NoTargetAnnotation
import firrtl.options.CustomFileEmission
import firrtl.options.Viewer.view
import firrtl.proto.ToProto
import firrtl.stage.{FirrtlOptions, Forms}
import firrtl.stage.TransformManager.TransformDependency
import firrtl.traversals.Foreachers._
import java.io.{ByteArrayOutputStream, Writer}
import scala.collection.mutable.ArrayBuffer
import Utils.{collectInstantiatedModules, throwInternalError}

/** This object defines Annotations that are used by Protocol Buffer emission.
  */
object Annotation {

  /** This will cause the enclosed circuit to be serialized to a Protocol Buffer with the provided suffix. */
  case class ProtoBufSerialization(circuit: ir.Circuit, suffix: Option[String])
      extends NoTargetAnnotation
      with CustomFileEmission {

    /** The hash code is cached for this case class because it includes a very large circuit operand whose hash is expensive
      * to compute.  This is safe to do because the case class is immutable.
      */
    override lazy val hashCode = scala.util.hashing.MurmurHash3.productHash(this)

    override protected def baseFileName(annotations: AnnotationSeq): String = {
      view[FirrtlOptions](annotations).outputFileName.getOrElse(circuit.main)
    }

    override def getBytes: Iterable[Byte] = {
      val ostream = new java.io.ByteArrayOutputStream
      ToProto.writeToStream(ostream, circuit)
      ostream.toByteArray()
    }
  }

}

/** Running this transform will cause a circuit to be emitted to ProtoBuf after some prerequisites have been
  * satisfied.
  *
  * This is not intended to be used directly.  Instead see one of the concrete emitters, e.g., [[Emitter.Low]].
  * @see [[Emitter.Chirrtl]]
  * @see [[Emitter.MHigh]]
  * @see [[Emitter.High]]
  * @see [[Emitter.Middle]]
  * @see [[Emitter.Low]]
  * @see [[Emitter.OptLow]]
  */
sealed abstract class ProtoBufEmitter(prereqs: Seq[TransformDependency])
    extends Transform
    with DependencyAPIMigration
    with firrtl.Emitter {

  override def prerequisites = prereqs
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf = Seq.empty
  override def invalidates(a: Transform) = false

  private def emitAllModules(circuit: Circuit): Seq[Annotation.ProtoBufSerialization] = {
    val modMap = circuit.modules.map(m => m.name -> m).toMap
    // Turn each module into it's own circuit with it as the top and all instantied modules as ExtModules
    circuit.modules.collect {
      case m: Module =>
        val instModules = collectInstantiatedModules(m, modMap)
        val extModules = instModules.map {
          case Module(info, name, ports, _) => ExtModule(info, name, ports, name, Seq.empty)
          case ext: ExtModule => ext
        }
        val newCircuit = Circuit(m.info, extModules :+ m, m.name)
        Annotation.ProtoBufSerialization(newCircuit, Some(outputSuffix))
    }
  }

  override def execute(state: CircuitState) = {
    val newAnnos = state.annotations.flatMap {
      case EmitCircuitAnnotation(a) if this.getClass == a =>
        Seq(
          Annotation.ProtoBufSerialization(state.circuit, Some(outputSuffix))
        )
      case EmitAllModulesAnnotation(a) if this.getClass == a =>
        emitAllModules(state.circuit)
      case _ => Seq()
    }
    state.copy(annotations = newAnnos ++ state.annotations)
  }

  def emit(state: CircuitState, writer: Writer): Unit = {
    val ostream = new java.io.ByteArrayOutputStream
    ToProto.writeToStream(ostream, state.circuit)
    writer.write(ostream.toString())
  }
}

/** This object defines different emitters that can be used to generate a Protocol Buffer of a FIRRTL circuit. */
object Emitter {

  /** Emit a FIRRTL circuit as ProtoBuf in CHIRRTL form. */
  class Chirrtl extends ProtoBufEmitter(Forms.ChirrtlForm) {
    override def outputSuffix = ".pb"
  }

  /** Emit a FIRRTL circuit as ProtoBuf in minimal High FIRRTL form.
    *
    * This will only have CHIRRTL constructs removed.  The circuit will not be deduplicated nor have widths inferred.
    * @see [[High]]
    */
  class MHigh extends ProtoBufEmitter(Forms.MinimalHighForm) {
    override def outputSuffix = ".mhi.pb"
  }

  /** Emit a FIRRTL circuit as ProtoBuf in High FIRRTL form.
    *
    * The emitted circuit will be structurally deduplicated and have widths inferred.
    * @see [[MHigh]]
    */
  class High extends ProtoBufEmitter(Forms.HighForm) {
    override def outputSuffix = ".hi.pb"
  }

  /** Emit a FIRRTL circuit as ProtoBuf in Mid FIRRTL form. */
  class Middle extends ProtoBufEmitter(Forms.MidForm) {
    override def outputSuffix = ".mid.pb"
  }

  /** Emit a FIRRTL circuit as ProtoBuf in Low FIRRTL form without optimizations.
    *
    * @see [[OptLow]]
    */
  class Low extends ProtoBufEmitter(Forms.LowForm) {
    override def outputSuffix = ".lo.pb"
  }

  /** Emit a FIRRTL circuit as ProtoBuf in Low FIRRTL form with optimizations.
    *
    * @see [[OptLow]]
    */
  class OptLow extends ProtoBufEmitter(Forms.LowFormOptimized) {
    override def outputSuffix = ".lo.pb"
  }

}
