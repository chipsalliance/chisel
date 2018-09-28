// See LICENSE for license details.

package firrtlTests.options

import org.scalatest._

import firrtl.options.Shell

class ShellSpec extends FlatSpec with Matchers {

  behavior of "Shell"

  it should "detect all registered libraries and transforms" in {
    val shell = new Shell("foo")

    info("Found FooTransform")
    shell.registeredTransforms.map(_.getClass.getName) should contain ("firrtlTests.options.FooTransform")

    info("Found BarLibrary")
    shell.registeredLibraries.map(_.getClass.getName) should contain ("firrtlTests.options.BarLibrary")
  }
}
