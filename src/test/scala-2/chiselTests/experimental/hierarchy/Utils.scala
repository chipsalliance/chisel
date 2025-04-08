// SPDX-License-Identifier: Apache-2.0

package chiselTests.experimental.hierarchy

import firrtl.annotations.{InstanceTarget, ModuleTarget, ReferenceTarget, Target}

trait Utils {
  // TODO promote to standard API (in FIRRTL) and perhaps even implement with a macro
  implicit class Str2RefTarget(str: String) {
    def rt: ReferenceTarget = Target.deserialize(str).asInstanceOf[ReferenceTarget]
    def it: InstanceTarget = Target.deserialize(str).asInstanceOf[InstanceTarget]
    def mt: ModuleTarget = Target.deserialize(str).asInstanceOf[ModuleTarget]
  }
}
