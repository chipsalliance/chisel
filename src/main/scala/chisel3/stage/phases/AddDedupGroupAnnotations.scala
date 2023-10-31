package chisel3.stage.phases

import firrtl.AnnotationSeq
import firrtl.options.{Dependency, Phase}
import firrtl.options.Viewer.view
import firrtl.transforms.DedupGroupAnnotation
import scala.annotation.nowarn

import chisel3.experimental.BaseModule
import chisel3.stage._
import chisel3.stage.CircuitSerializationAnnotation._
import chisel3.ChiselException
import chisel3.experimental.dedupGroup
import chisel3.internal.firrtl.{DefBlackBox, DefClass, DefIntrinsicModule}

class AddDedupGroupAnnotations extends Phase {
  override def prerequisites = Seq.empty
  override def optionalPrerequisites: Seq[Dependency[Phase]] = Seq(Dependency[Convert])
  override def optionalPrerequisiteOf = Seq.empty
  override def invalidates(a: Phase) = false

  def transform(annotations: AnnotationSeq): AnnotationSeq = {
    val chiselOptions = view[ChiselOptions](annotations)
    val circuit = chiselOptions.chiselCircuit.getOrElse {
      throw new ChiselException(
        s"Unable to locate the elaborated circuit, did ${classOf[Elaborate].getName} run correctly"
      )
    }

    val skipAnnos = annotations.collect { case x: DedupGroupAnnotation => x.target }.toSet

    @nowarn("msg=class Port")
    val annos = circuit.components.filter {
      case x @ DefBlackBox(id, _, _, _, _)   => !id._isImportedDefinition
      case DefIntrinsicModule(_, _, _, _, _) => false
      case DefClass(_, _, _, _)              => false
      case x                                 => true
    }.collect {
      case x if !(skipAnnos.contains(x.id.toTarget)) => DedupGroupAnnotation(x.id.toTarget, x.id.desiredName)
    }
    annos ++ annotations
  }
}
