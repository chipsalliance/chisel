// SPDX-License-Identifier: Apache-2.0

package firrtlTests.stage.phases

import firrtl.annotations.Annotation
import firrtl.stage.phases.AddDefaults
import firrtl.transforms.BlackBoxTargetDirAnno
import firrtl.stage.InfoModeAnnotation
import firrtl.options.{Dependency, Phase, TargetDirAnnotation}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class AddDefaultsSpec extends AnyFlatSpec with Matchers {

  class Fixture { val phase: Phase = new AddDefaults }

  behavior.of(classOf[AddDefaults].toString)

  it should "add expected default annotations and nothing else" in new Fixture {
    val expected = Seq(
      (a: Annotation) => a match { case BlackBoxTargetDirAnno(b) => b == TargetDirAnnotation().directory },
      (a: Annotation) => a match { case InfoModeAnnotation(b) => b == InfoModeAnnotation().modeName }
    )

    phase.transform(Seq.empty).zip(expected).map { case (x, f) => f(x) should be(true) }
  }

  it should "not overwrite existing annotations" in new Fixture {
    val input = Seq(
      BlackBoxTargetDirAnno("foo"),
      InfoModeAnnotation("ignore")
    )

    phase.transform(input).toSeq should be(input)
  }
}
