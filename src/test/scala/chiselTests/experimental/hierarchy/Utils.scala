// SPDX-License-Identifier: Apache-2.0

package chiselTests.experimental.hierarchy

import _root_.firrtl.annotations._
import chiselTests.ChiselRunners
import org.scalatest.matchers.should.Matchers

trait Utils extends ChiselRunners with chiselTests.Utils with Matchers {
  // TODO promote to standard API (in FIRRTL) and perhaps even implement with a macro
  implicit class Str2RefTarget(str: String) {
    def rt: ReferenceTarget = Target.deserialize(str).asInstanceOf[ReferenceTarget]
    def it: InstanceTarget = Target.deserialize(str).asInstanceOf[InstanceTarget]
    def mt: ModuleTarget = Target.deserialize(str).asInstanceOf[ModuleTarget]
    def ct: CircuitTarget = Target.deserialize(str).asInstanceOf[CircuitTarget]
  }
}
