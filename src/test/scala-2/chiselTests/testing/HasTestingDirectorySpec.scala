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

  behavior of ("HasTestingDirectory.timestamp")

  it should ("return the same directory for a given instance of the type class") in {

    val foo = HasTestingDirectory.timestamp

    foo.getDirectory should be(foo.getDirectory)

  }

  it should ("return different directories for separate instances of the type class") in {

    val foo = HasTestingDirectory.timestamp
    val bar = HasTestingDirectory.timestamp

    foo.getDirectory should not be (bar.getDirectory)

  }

  behavior of ("HasTestingDirectory.default")

  it should ("return different directories for different functions that require a type class implementation") in {

    def foo(implicit testingDirectory: HasTestingDirectory) = {
      testingDirectory.getDirectory
    }

    foo should not be (foo)

  }

  it should ("allow the user to force the same directory by creating an implicit val") in {

    def foo(implicit testingDirectory: HasTestingDirectory) = {
      testingDirectory.getDirectory
    }

    implicit val bar = implicitly[HasTestingDirectory]

    foo should be(foo)

  }

}
