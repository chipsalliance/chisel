// See LICENSE for license details.

package firrtlTests.annotationTests

import firrtl.annotations.{CircuitTarget, GenericTarget, ModuleTarget, Target}
import firrtl.annotations.TargetToken._
import firrtlTests.FirrtlPropSpec

class TargetSpec extends FirrtlPropSpec {
  def check(comp: Target): Unit = {
    val named = Target.convertTarget2Named(comp)
    println(named)
    val comp2 = Target.convertNamed2Target(named)
    assert(comp.toGenericTarget.complete == comp2)
  }
  property("Serialization of Targets should work") {
    val circuit = CircuitTarget("Circuit")
    val top = circuit.module("Top")
    val targets: Seq[(Target, String)] =
      Seq(
        (circuit, "~Circuit"),
        (top, "~Circuit|Top"),
        (top.instOf("i", "I"), "~Circuit|Top/i:I"),
        (top.ref("r"), "~Circuit|Top>r"),
        (top.ref("r").index(1).field("hi").clock, "~Circuit|Top>r[1].hi@clock"),
        (GenericTarget(None, None, Vector(Ref("r"))), "~???|???>r")
      )
    targets.foreach { case (t, str) =>
      assert(t.serialize == str, s"$t does not properly serialize")
    }
  }
  property("Should convert to/from Named") {
    check(Target(Some("Top"), None, Nil))
    check(Target(Some("Top"), Some("Top"), Nil))
    check(Target(Some("Top"), Some("Other"), Nil))
    val r1 = Seq(Ref("r1"), Field("I"))
    val r2 = Seq(Ref("r2"), Index(0))
    check(Target(Some("Top"), Some("Top"), r1))
    check(Target(Some("Top"), Some("Top"), r2))
  }
  property("Should enable creating from API") {
    val top = ModuleTarget("Top","Top")
    val x_reg0_data = top.instOf("x", "X").ref("reg0").field("data")
    top.instOf("x", "x")
    top.ref("y")
    println(x_reg0_data)
  }
  property("Should serialize and deserialize") {
    val circuit = CircuitTarget("Circuit")
    val top = circuit.module("Top")
    val targets: Seq[Target] =
      Seq(circuit, top, top.instOf("i", "I"), top.ref("r"),
        top.ref("r").index(1).field("hi").clock, GenericTarget(None, None, Vector(Ref("r"))))
    targets.foreach { t =>
      assert(Target.deserialize(t.serialize) == t, s"$t does not properly serialize/deserialize")
    }
  }
  property("Pretty Printer should work") {
    val circuit = CircuitTarget("A")
    val top = circuit.module("B")
    val targets = Seq(
      (circuit, "circuit A:"),
      (top,
       """|circuit A:
          |└── module B:""".stripMargin),
      (top.instOf("c", "C"),
       """|circuit A:
          |└── module B:
          |    └── inst c of C:""".stripMargin),
      (top.ref("r"),
       """|circuit A:
          |└── module B:
          |    └── r""".stripMargin),
      (top.ref("r").index(1).field("hi").clock,
       """|circuit A:
          |└── module B:
          |    └── r[1].hi@clock""".stripMargin),
      (GenericTarget(None, None, Vector(Ref("r"))),
       """|circuit ???:
          |└── module ???:
          |    └── r""".stripMargin)
    )
    targets.foreach { case (t, str) => assert(t.prettyPrint() == str, s"$t didn't properly prettyPrint") }
  }
}
