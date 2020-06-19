// See LICENSE for license details.

package firrtl.stage.phases

import firrtl.{AnnotationSeq, EmitAnnotation, EmitCircuitAnnotation}
import firrtl.stage.{CompilerAnnotation, RunFirrtlTransformAnnotation}
import firrtl.options.{Dependency, Phase}

/** [[firrtl.options.Phase Phase]] that adds a [[firrtl.EmitCircuitAnnotation EmitCircuitAnnotation]] derived from a
  * [[firrtl.stage.CompilerAnnotation CompilerAnnotation]] if one does not already exist.
  */
class AddImplicitEmitter extends Phase {

  override def prerequisites = Seq(Dependency[AddDefaults])

  override def optionalPrerequisiteOf = Seq.empty

  override def invalidates(a: Phase) = false

  def transform(annos: AnnotationSeq): AnnotationSeq = {
    val emitter = annos.collectFirst{ case a: EmitAnnotation => a }
    val compiler = annos.collectFirst{ case CompilerAnnotation(a) => a }

    if (emitter.isEmpty && compiler.nonEmpty) {
      annos.flatMap{
        case a: CompilerAnnotation => Seq(a,
                                          RunFirrtlTransformAnnotation(compiler.get.emitter),
                                          EmitCircuitAnnotation(compiler.get.emitter.getClass))
        case a => Some(a)
      }
    } else {
      annos
    }
  }

}
