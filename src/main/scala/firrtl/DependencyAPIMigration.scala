// SPDX-License-Identifier: Apache-2.0

package firrtl

import firrtl.stage.TransformManager.TransformDependency

/** This trait helps ease migration from old [[firrtl.CircuitForm CircuitForm]] specification of dependencies to
  * Dependency API specification of dependencies. This trait implements deprecated, abstract [[Transform]] methods
  * (`inputForm` and `outputForm`) for you and sets default values for dependencies:
  *
  *    - `prerequisites` are empty
  *    - `optionalPrerequisites` are empty
  *    - `optionalPrerequisiteOf` are empty
  *    - all transforms are invalidated
  *
  * For more information, see: https://bit.ly/2Voppre
  */
@deprecated(
  "Please remove DependencyAPIMigration from your Transform to be compatible with FIRRTL 1.6.",
  "FIRRTL 1.5"
)
trait DependencyAPIMigration { this: Transform =>

  @deprecated("Use Dependency API methods for equivalent functionality. See: https://bit.ly/2Voppre", "FIRRTL 1.3")
  final override def inputForm: CircuitForm = UnknownForm

  @deprecated("Use Dependency API methods for equivalent functionality. See: https://bit.ly/2Voppre", "FIRRTL 1.3")
  final override def outputForm: CircuitForm = UnknownForm

  override def prerequisites: Seq[TransformDependency] = Seq.empty

  override def optionalPrerequisites: Seq[TransformDependency] = Seq.empty

  override def optionalPrerequisiteOf: Seq[TransformDependency] = Seq.empty

  override def invalidates(a: Transform): Boolean = true

}
