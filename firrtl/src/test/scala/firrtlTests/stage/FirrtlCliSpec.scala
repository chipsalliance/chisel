// SPDX-License-Identifier: Apache-2.0

package firrtlTests.stage

import firrtl.stage.RunFirrtlTransformAnnotation
import firrtl.options.Shell
import firrtl.stage.FirrtlCli
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class FirrtlCliSpec extends AnyFlatSpec with Matchers {

  behavior.of("FirrtlCli for RunFirrtlTransformAnnotation / -fct / --custom-transforms")

  it should "preserver transform order" in {
    val shell = new Shell("foo") with FirrtlCli
    val args = Array(
      "--custom-transforms",
      "firrtl.transforms.CombineCats"
    )
    val expected = Seq(
      classOf[firrtl.transforms.CombineCats]
    )

    shell
      .parse(args)
      .collect { case a: RunFirrtlTransformAnnotation => a }
      .zip(expected)
      .map { case (RunFirrtlTransformAnnotation(a), b) => a.getClass should be(b) }
  }

}
