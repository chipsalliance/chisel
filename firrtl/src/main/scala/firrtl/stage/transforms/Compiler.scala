// SPDX-License-Identifier: Apache-2.0

package firrtl.stage.transforms

import firrtl.options.DependencyManagerUtils.CharSet
import firrtl.stage.TransformManager
import firrtl.Transform

/** A [[firrtl.stage.TransformManager TransformManager]] of
  */
class Compiler(
  targets:      Seq[TransformManager.TransformDependency],
  currentState: Seq[TransformManager.TransformDependency] = Seq.empty,
  knownObjects: Set[Transform] = Set.empty)
    extends TransformManager(targets, currentState, knownObjects) {

  override val wrappers = Seq(
    (a: Transform) => ExpandPrepares(a),
    (a: Transform) => CatchCustomTransformExceptions(a),
    (a: Transform) => UpdateAnnotations(a)
  )

}
