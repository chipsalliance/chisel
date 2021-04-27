// SPDX-License-Identifier: Apache-2.0

package firrtl
package passes
package memlib

import firrtl.stage.Forms

class DumpMemoryAnnotations extends Transform with DependencyAPIMigration {

  override def prerequisites = Forms.MidForm
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf = Forms.MidEmitters
  override def invalidates(a: Transform) = false

  def execute(state: CircuitState): CircuitState = {
    state.copy(annotations = state.annotations.flatMap {
      // convert and remove AnnotatedMemoriesAnnotation to CustomFileEmission
      case AnnotatedMemoriesAnnotation(annotatedMemories) =>
        state.annotations.collect {
          case a: MemLibOutConfigFileAnnotation =>
            a.copy(annotatedMemories = annotatedMemories)
          // todo convert xxx to verilogs here.
        }
      case MemLibOutConfigFileAnnotation(_, Nil) => Nil
      case a                                     => Seq(a)
    })
  }
}
