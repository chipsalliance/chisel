// SPDX-License-Identifier: Apache-2.0

package firrtl.stage.phases

import firrtl.{AnnotationSeq, EmitAnnotation, EmitCircuitAnnotation, Emitter}
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
    val emit = annos.collectFirst { case a: EmitAnnotation => a }
    val emitter = annos.collectFirst { case RunFirrtlTransformAnnotation(e: Emitter) => e }
    val compiler = annos.collectFirst { case CompilerAnnotation(a) => a }

    if (emit.isEmpty && (compiler.nonEmpty || emitter.nonEmpty)) {
      annos.flatMap {
        case a: CompilerAnnotation =>
          Seq(
            a,
            RunFirrtlTransformAnnotation(compiler.get.emitter),
            EmitCircuitAnnotation(compiler.get.emitter.getClass)
          )
        case a @ RunFirrtlTransformAnnotation(e: Emitter) => Seq(a, EmitCircuitAnnotation(e.getClass))
        case a => Some(a)
      }
    } else {
      annos
    }
  }

}
