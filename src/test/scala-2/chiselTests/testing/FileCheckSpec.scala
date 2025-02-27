// SPDX-License-Identifier: Apache-2.0

package chiselTests.testing

import chisel3.testing.FileCheck
import chisel3.testing.scalatest.TestingDirectory
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class FileCheckSpec extends AnyFlatSpec with Matchers with TestingDirectory with FileCheck {

  behavior of ("FileCheck")

  it should "check a string showing success" in {
    "Hello world!".fileCheck()(
      """|CHECK:      Hello
         |CHECK-SAME: world
         |""".stripMargin
    )
    "Hello world!".fileCheck()("CHECK: Hello world!")
  }

  it should "check a string showing failure" in {
    intercept[FileCheck.Exceptions.NonZeroExitCode] {
      "Hello world!".fileCheck()("CHECK: no match")
    }
  }

  it should "allow for a user to pass additional options" in {
    "Hello world!".fileCheck("--check-prefix=FOO")("FOO:Hello world!")
  }

}
