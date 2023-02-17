// SPDX-License-Identifier: Apache-2.0

package firrtl.stage

import firrtl.{CircuitState, DependencyAPIMigration, Transform}
import firrtl.options.{Dependency, DependencyManager}

object TransformManager {

  /** The type used to represent dependencies between [[Transform]]s */
  type TransformDependency = Dependency[Transform]

}
