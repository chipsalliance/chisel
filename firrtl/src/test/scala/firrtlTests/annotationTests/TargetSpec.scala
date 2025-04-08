// SPDX-License-Identifier: Apache-2.0

package firrtlTests.annotationTests

import firrtl.annotations.{GenericTarget, ModuleTarget, Target}
import firrtl.annotations.TargetToken._
import firrtl.testutils.FirrtlPropSpec

class TargetSpec extends FirrtlPropSpec {
  def check(comp: Target): Unit = {
    val named = Target.convertTarget2Named(comp)
    val comp2 = Target.convertNamed2Target(named)
    assert(comp.toGenericTarget.complete == comp2)
  }
  property("Serialization of Targets should work") {
    val top = ModuleTarget("Top")
    val targets: Seq[(Target, String)] =
      Seq(
        (top, "~|Top"),
        (top.instOf("i", "I"), "~|Top/i:I"),
        (top.ref("r"), "~|Top>r"),
        (top.ref("r").index(1).field("hi"), "~|Top>r[1].hi"),
        (GenericTarget(None, Vector(Ref("r"))), "~|???>r")
      )
    targets.foreach { case (t, str) =>
      assert(t.serialize == str, s"$t does not properly serialize")
    }
  }
  property("Should convert to/from Named") {
    check(Target(Some("Top"), Nil))
    check(Target(Some("Other"), Nil))
    val r1 = Seq(Ref("r1"), Field("I"))
    val r2 = Seq(Ref("r2"), Index(0))
    check(Target(Some("Top"), r1))
    check(Target(Some("Top"), r2))
  }
  property("Should enable creating from API") {
    val top = ModuleTarget("Top")
    val x_reg0_data = top.instOf("x", "X").ref("reg0").field("data")
    top.instOf("x", "x")
    top.ref("y")
  }
  property("Should serialize and deserialize") {
    val top = ModuleTarget("Top")
    val targets: Seq[Target] =
      Seq(
        top,
        top.instOf("i", "I"),
        top.ref("r"),
        top.ref("r").index(1).field("hi"),
        GenericTarget(None, Vector(Ref("r")))
      )
    targets.foreach { t =>
      assert(Target.deserialize(t.serialize) == t, s"$t does not properly serialize/deserialize")
    }
  }
}
