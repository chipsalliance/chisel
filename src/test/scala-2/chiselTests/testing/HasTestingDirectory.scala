// SPDX-License-Identifier: Apache-2.0

package chiselTests.testing

import chisel3.testing.HasTestingDirectory
import java.nio.file.Paths
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class HasTestingDirectorySpec extends AnyFlatSpec with Matchers {

  val foo = new HasTestingDirectory {
    override def getDirectory = Paths.get("foo", "bar")
  }

  behavior of ("HasTestingDirectory.getDirectory")

  it should ("return the specified output directory") in {

    foo.getDirectory should be(Paths.get("foo", "bar"))

  }

  behavior of ("HasTestingDirectory.withSubdirectory")

  it should ("return a subdirectory of the original directory") in {

    val baz = foo.withSubdirectory("baz")

    baz.getDirectory should be(Paths.get("foo", "bar", "baz"))

  }

}
