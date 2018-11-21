// See LICENSE for license details.

package firrtlTests.options

import org.scalatest.{FlatSpec, Matchers}

import firrtl.annotations.NoTargetAnnotation
import firrtl.options.Shell

class ShellSpec extends FlatSpec with Matchers {

  case object A extends NoTargetAnnotation
  case object B extends NoTargetAnnotation
  case object C extends NoTargetAnnotation
  case object D extends NoTargetAnnotation
  case object E extends NoTargetAnnotation

  trait AlphabeticalCli { this: Shell =>
    parser.opt[Unit]('c', "c-option").unbounded().action( (x, c) => C +: c )
    parser.opt[Unit]('d', "d-option").unbounded().action( (x, c) => D +: c )
    parser.opt[Unit]('e', "e-option").unbounded().action( (x, c) => E +: c ) }

  behavior of "Shell"

  it should "detect all registered libraries and transforms" in {
    val shell = new Shell("foo")

    info("Found FooTransform")
    shell.registeredTransforms.map(_.getClass.getName) should contain ("firrtlTests.options.FooTransform")

    info("Found BarLibrary")
    shell.registeredLibraries.map(_.getClass.getName) should contain ("firrtlTests.options.BarLibrary")
  }

  it should "correctly order annotations and options" in {
    val shell = new Shell("foo") with AlphabeticalCli

    shell.parse(Array("-c", "-d", "-e"), Seq(A, B)).toSeq should be (Seq(A, B, C, D, E))
  }
}
