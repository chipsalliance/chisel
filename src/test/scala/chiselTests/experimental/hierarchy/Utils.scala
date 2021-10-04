// SPDX-License-Identifier: Apache-2.0

package chiselTests.experimental.hierarchy

import chisel3._
import _root_.firrtl.annotations._
import chisel3.stage.{ChiselCircuitAnnotation, CircuitSerializationAnnotation, DesignAnnotation}
import chiselTests.ChiselRunners
import firrtl.stage.FirrtlCircuitAnnotation
import org.scalatest.matchers.should.Matchers

trait Utils extends ChiselRunners with chiselTests.Utils with Matchers {
  import Annotations._
  // TODO promote to standard API (in FIRRTL) and perhaps even implement with a macro
  implicit class Str2RefTarget(str: String) {
    def rt: ReferenceTarget = Target.deserialize(str).asInstanceOf[ReferenceTarget]
    def it: InstanceTarget = Target.deserialize(str).asInstanceOf[InstanceTarget]
    def mt: ModuleTarget = Target.deserialize(str).asInstanceOf[ModuleTarget]
    def ct: CircuitTarget = Target.deserialize(str).asInstanceOf[CircuitTarget]
  }
}
