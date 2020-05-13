// See LICENSE for license details.

package firrtlTests.features

import firrtl.features.{LowerCaseNames, UpperCaseNames}

import firrtl.{CircuitState, Parser}
import firrtl.annotations.CircuitTarget

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class LetterCaseTransformSpec extends AnyFlatSpec with Matchers {

  class CircuitFixture {
    private val input =
      """|circuit Foo:
         |  module Foo:
         |    node Bar = UInt<1>(0)
         |    node baz = UInt<1>(0)
         |    node QUX = UInt<1>(0)
         |    node quuxQuux = UInt<1>(0)
         |    node QuuzQuuz = UInt<1>(0)
         |    node corge_corge = UInt<1>(0)
         |""".stripMargin
    val state = CircuitState(Parser.parse(input), Seq.empty)
  }

  behavior of "LowerCaseNames"

  it should "change all names to lowercase" in new CircuitFixture {
    val string = (new LowerCaseNames).execute(state).circuit.serialize
    List("foo", "bar", "baz", "qux", "quuxquux", "quuzquuz", "corge_corge").foreach(string should include (_))
  }

  behavior of "UpperCaseNames"

  it should "change all names to uppercase" in new CircuitFixture {
    val string = (new UpperCaseNames).execute(state).circuit.serialize
    List("FOO", "BAR", "BAZ", "QUX", "QUUXQUUX", "QUUZQUUZ", "CORGE_CORGE").foreach(string should include (_))
  }

}
