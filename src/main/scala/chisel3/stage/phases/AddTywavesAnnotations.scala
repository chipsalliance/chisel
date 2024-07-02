package chisel3.stage.phases

import chisel3.stage.ChiselCircuitAnnotation
import chisel3.tywaves.TywavesChiselAnnotation
import firrtl.AnnotationSeq
import firrtl.options.{Dependency, Phase}

// TODO: this phase should be added to the ChiselStage such that it is executed before
//  Convert phase and run only if Tywaves is enabled. Otherwise, it should be skipped
//  since it adds unnecessary annotations for other purposes.

/** This phase adds annotations to the Chisel circuit that will be propagated to FIRRTL and
  * used by firtool to generate proper debug file format for Tywaves.
  *
  *   - It parses the whole Chisel circuit
  *   - Extracts the necessary information from each component
  *   - Annotates each FIRRTL target
  */
class AddTywavesAnnotations extends Phase {

  override def prerequisites = Seq(Dependency[Elaborate])
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf = Seq(Dependency[MaybeInjectingPhase])
  override def invalidates(a: Phase) = false

  def transform(annotations: AnnotationSeq): AnnotationSeq = annotations.flatMap {
    case a: ChiselCircuitAnnotation =>
      // Generate the Tywaves annotations from the Chisel circuit
      val anno = TywavesChiselAnnotation.generate(a.circuit)
      val circuit = a.circuit.copy(annotations = a.circuit.annotations ++ anno)
//      Some(a) ++
      Some(ChiselCircuitAnnotation(circuit))
    case a => Some(a)
  }
}
