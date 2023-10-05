// SPDX-License-Identifier: Apache-2.0

package firrtlTests.options.phases

import firrtl.options.{Phase, TargetDirAnnotation}
import firrtl.options.phases.AddDefaults
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class AddDefaultsSpec extends AnyFlatSpec with Matchers {

  class Fixture {
    val phase: Phase = new AddDefaults
    val targetDir = TargetDirAnnotation("foo")
    val defaultDir = TargetDirAnnotation(".")
  }

  behavior.of(classOf[AddDefaults].toString)

  it should "add a TargetDirAnnotation if it does not exist" in new Fixture {
    phase.transform(Seq.empty).toSeq should be(Seq(defaultDir))
  }

  it should "don't add a TargetDirAnnotation if it exists" in new Fixture {
    phase.transform(Seq(targetDir)).toSeq should be(Seq(targetDir))
  }
}
