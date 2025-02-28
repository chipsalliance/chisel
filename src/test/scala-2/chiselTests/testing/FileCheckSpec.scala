// SPDX-License-Identifier: Apache-2.0

package chiselTests.testing.scalatest

import chisel3.testing.FileCheck.Exceptions
import chisel3.testing.scalatest.FileCheck
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class FileCheckSpec extends AnyFlatSpec with Matchers with FileCheck {

  behavior of ("FileCheck")

  it should "check an input string and suceed" in {
    "Hello world!".fileCheck()(
      """|CHECK:      Hello
         |CHECK-SAME: world
         |""".stripMargin
    )
    "Hello world!".fileCheck()("CHECK: Hello world!")
  }

  it should "throw a NonZeroExitCode exception on failure" in {
    intercept[Exceptions.NonZeroExitCode] {
      "Hello world!".fileCheck()("CHECK: no match")
    }
  }

  it should "allow for a user to pass additional options" in {
    "Hello world!".fileCheck("--check-prefix=FOO")("FOO:Hello world!")
  }

}
