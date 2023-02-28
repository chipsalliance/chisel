// SPDX-License-Identifier: Apache-2.0

package firrtlTests.annotationTests

import firrtl.analyses.InstanceKeyGraph.InstanceKey
import firrtl.annotations._
import firrtl.annotations.TargetToken._
import firrtl.annotations.TargetUtils._
import firrtl.testutils.FirrtlFlatSpec

class TargetUtilsSpec extends FirrtlFlatSpec {

  behavior.of("instKeyPathToTarget")

  it should "create a ModuleTarget for the top module" in {
    val input = InstanceKey("Top", "Top") :: Nil
    val expected = ModuleTarget("Top", "Top")
    instKeyPathToTarget(input) should be(expected)
  }

  it should "create absolute InstanceTargets" in {
    val input = InstanceKey("Top", "Top") ::
      InstanceKey("foo", "Foo") ::
      InstanceKey("bar", "Bar") ::
      Nil
    val expected = InstanceTarget("Top", "Top", Seq((Instance("foo"), OfModule("Foo"))), "bar", "Bar")
    instKeyPathToTarget(input) should be(expected)
  }

  it should "support starting somewhere down the path" in {
    val input = InstanceKey("Top", "Top") ::
      InstanceKey("foo", "Foo") ::
      InstanceKey("bar", "Bar") ::
      InstanceKey("fizz", "Fizz") ::
      Nil
    val expected = InstanceTarget("Top", "Bar", Seq(), "fizz", "Fizz")
    instKeyPathToTarget(input, Some("Bar")) should be(expected)
  }

  behavior.of("unfoldInstanceTargets")

  it should "return nothing for ModuleTargets" in {
    val input = ModuleTarget("Top", "Foo")
    unfoldInstanceTargets(input) should be(Iterable())
  }

  it should "return all other InstanceTargets to the same instance" in {
    val input = ModuleTarget("Top", "Top").instOf("foo", "Foo").instOf("bar", "Bar").instOf("fizz", "Fizz")
    val expected =
      input ::
        ModuleTarget("Top", "Foo").instOf("bar", "Bar").instOf("fizz", "Fizz") ::
        ModuleTarget("Top", "Bar").instOf("fizz", "Fizz") ::
        Nil
    unfoldInstanceTargets(input) should be(expected)
  }
}
