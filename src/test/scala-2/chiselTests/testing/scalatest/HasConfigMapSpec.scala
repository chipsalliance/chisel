// SPDX-License-Identifier: Apache-2.0

package chiselTests.testing.scalatest

import chisel3.testing.scalatest.HasConfigMap
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class HasConfigMapSpec extends AnyFlatSpec with Matchers with HasConfigMap {

  behavior of "HasConfigMap"

  it should "be accessible inside a test" in {
    configMap
  }

  val outsideTestException = intercept[Exception] {
    configMap
  }

  it should "throw a RuntimeException if used outside a test" in {
    outsideTestException shouldBe a[RuntimeException]
    outsideTestException.getMessage should include("configMap may only be accessed inside a Scalatest test")
  }

}
