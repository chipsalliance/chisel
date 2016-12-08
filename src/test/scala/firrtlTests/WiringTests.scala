// See LICENSE for license details.

package firrtlTests

import java.io._
import org.scalatest._
import org.scalatest.prop._
import firrtl._
import firrtl.ir.Circuit
import firrtl.passes._
import firrtl.Parser.IgnoreInfo
import annotations._
import wiring.WiringUtils._
import wiring._

class WiringTests extends FirrtlFlatSpec {
  private def executeTest(input: String, expected: Seq[String], passes: Seq[Pass]) = {
    val c = passes.foldLeft(Parser.parse(input.split("\n").toIterator)) {
      (c: Circuit, p: Pass) => p.run(c)
    }
    val lines = c.serialize.split("\n") map normalized

    expected foreach { e =>
      lines should contain(e)
    }
  }

  def passes = Seq(
    ToWorkingIR,
    ResolveKinds,
    InferTypes,
    ResolveGenders,
    InferWidths
  )

  "Wiring from r to X" should "work" in {
    val sinks = Set("X")
    val sas = WiringInfo("C", "r", sinks, "pin", "A")
    val input =
      """circuit Top :
        |  module Top :
        |    input clock: Clock
        |    inst a of A
        |    a.clock <= clock
        |  module A :
        |    input clock: Clock
        |    inst b of B
        |    b.clock <= clock
        |    inst x of X
        |    x.clock <= clock
        |    inst d of D
        |    d.clock <= clock
        |  module B :
        |    input clock: Clock
        |    inst c of C
        |    c.clock <= clock
        |    inst d of D
        |    d.clock <= clock
        |  module C :
        |    input clock: Clock
        |    reg r: UInt<5>, clock
        |  module D :
        |    input clock: Clock
        |    inst x1 of X
        |    x1.clock <= clock
        |    inst x2 of X
        |    x2.clock <= clock
        |  extmodule X :
        |    input clock: Clock
        |""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    input clock: Clock
        |    inst a of A
        |    a.clock <= clock
        |  module A :
        |    input clock: Clock
        |    inst b of B
        |    b.clock <= clock
        |    inst x of X
        |    x.clock <= clock
        |    inst d of D
        |    d.clock <= clock
        |    wire r: UInt<5>
        |    r <= b.r
        |    x.pin <= r
        |    d.r <= r
        |  module B :
        |    input clock: Clock
        |    output r: UInt<5>
        |    inst c of C
        |    c.clock <= clock
        |    inst d of D
        |    d.clock <= clock
        |    r <= c.r_0
        |    d.r <= r
        |  module C :
        |    input clock: Clock
        |    output r_0: UInt<5>
        |    reg r: UInt<5>, clock
        |    r_0 <= r
        |  module D :
        |    input clock: Clock
        |    input r: UInt<5>
        |    inst x1 of X
        |    x1.clock <= clock
        |    inst x2 of X
        |    x2.clock <= clock
        |    x1.pin <= r
        |    x2.pin <= r
        |  extmodule X :
        |    input clock: Clock
        |    input pin: UInt<5>
        |""".stripMargin
    val c = passes.foldLeft(parse(input)) {
      (c: Circuit, p: Pass) => p.run(c)
    }
    val wiringPass = new Wiring(Seq(sas))
    val retC = wiringPass.run(c)
    (parse(retC.serialize).serialize) should be (parse(check).serialize)
  }

  "Wiring from r.x to X" should "work" in {
    val sinks = Set("X")
    val sas = WiringInfo("A", "r.x", sinks, "pin", "A")
    val input =
      """circuit Top :
        |  module Top :
        |    input clock: Clock
        |    inst a of A
        |    a.clock <= clock
        |  module A :
        |    input clock: Clock
        |    reg r : {x: UInt<5>}, clock
        |    inst x of X
        |    x.clock <= clock
        |  extmodule X :
        |    input clock: Clock
        |""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    input clock: Clock
        |    inst a of A
        |    a.clock <= clock
        |  module A :
        |    input clock: Clock
        |    reg r: {x: UInt<5>}, clock
        |    inst x of X
        |    x.clock <= clock
        |    wire r_x: UInt<5>
        |    r_x <= r.x
        |    x.pin <= r_x
        |  extmodule X :
        |    input clock: Clock
        |    input pin: UInt<5>
        |""".stripMargin
    val c = passes.foldLeft(parse(input)) {
      (c: Circuit, p: Pass) => p.run(c)
    }
    val wiringPass = new Wiring(Seq(sas))
    val retC = wiringPass.run(c)
    (parse(retC.serialize).serialize) should be (parse(check).serialize)
  }
  "Wiring from clock to X" should "work" in {
    val sinks = Set("X")
    val sas = WiringInfo("A", "clock", sinks, "pin", "A")
    val input =
      """circuit Top :
        |  module Top :
        |    input clock: Clock
        |    inst a of A
        |    a.clock <= clock
        |  module A :
        |    input clock: Clock
        |    inst x of X
        |    x.clock <= clock
        |  extmodule X :
        |    input clock: Clock
        |""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    input clock: Clock
        |    inst a of A
        |    a.clock <= clock
        |  module A :
        |    input clock: Clock
        |    inst x of X
        |    x.clock <= clock
        |    wire clock_0: Clock
        |    clock_0 <= clock
        |    x.pin <= clock_0
        |  extmodule X :
        |    input clock: Clock
        |    input pin: Clock
        |""".stripMargin
    val c = passes.foldLeft(parse(input)) {
      (c: Circuit, p: Pass) => p.run(c)
    }
    val wiringPass = new Wiring(Seq(sas))
    val retC = wiringPass.run(c)
    (parse(retC.serialize).serialize) should be (parse(check).serialize)
  }
  "Two sources" should "fail" in {
    val sinks = Set("X")
    val sas = WiringInfo("A", "clock", sinks, "pin", "Top")
    val input =
      """circuit Top :
        |  module Top :
        |    input clock: Clock
        |    inst a1 of A
        |    a1.clock <= clock
        |    inst a2 of A
        |    a2.clock <= clock
        |  module A :
        |    input clock: Clock
        |    inst x of X
        |    x.clock <= clock
        |  extmodule X :
        |    input clock: Clock
        |""".stripMargin
    intercept[WiringException] {
      val c = passes.foldLeft(parse(input)) {
        (c: Circuit, p: Pass) => p.run(c)
      }
      val wiringPass = new Wiring(Seq(sas))
      val retC = wiringPass.run(c)
    }
  }
  "Wiring from A.clock to X, with 2 A's, and A as top" should "work" in {
    val sinks = Set("X")
    val sas = WiringInfo("A", "clock", sinks, "pin", "A")
    val input =
      """circuit Top :
        |  module Top :
        |    input clock: Clock
        |    inst a1 of A
        |    a1.clock <= clock
        |    inst a2 of A
        |    a2.clock <= clock
        |  module A :
        |    input clock: Clock
        |    inst x of X
        |    x.clock <= clock
        |  extmodule X :
        |    input clock: Clock
        |""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    input clock: Clock
        |    inst a1 of A
        |    a1.clock <= clock
        |    inst a2 of A
        |    a2.clock <= clock
        |  module A :
        |    input clock: Clock
        |    inst x of X
        |    x.clock <= clock
        |    wire clock_0: Clock
        |    clock_0 <= clock
        |    x.pin <= clock_0
        |  extmodule X :
        |    input clock: Clock
        |    input pin: Clock
        |""".stripMargin
    val c = passes.foldLeft(parse(input)) {
      (c: Circuit, p: Pass) => p.run(c)
    }
    val wiringPass = new Wiring(Seq(sas))
    val retC = wiringPass.run(c)
    (parse(retC.serialize).serialize) should be (parse(check).serialize)
  }
  "Wiring from A.clock to X, with 2 A's, and A as top, but Top instantiates X" should "error" in {
    val sinks = Set("X")
    val sas = WiringInfo("A", "clock", sinks, "pin", "A")
    val input =
      """circuit Top :
        |  module Top :
        |    input clock: Clock
        |    inst a1 of A
        |    a1.clock <= clock
        |    inst a2 of A
        |    a2.clock <= clock
        |    inst x of X
        |    x.clock <= clock
        |  module A :
        |    input clock: Clock
        |    inst x of X
        |    x.clock <= clock
        |  extmodule X :
        |    input clock: Clock
        |""".stripMargin
    intercept[WiringException] {
      val c = passes.foldLeft(parse(input)) {
        (c: Circuit, p: Pass) => p.run(c)
      }
      val wiringPass = new Wiring(Seq(sas))
      val retC = wiringPass.run(c)
    }
  }
  "Wiring from A.r[a] to X" should "work" in {
    val sinks = Set("X")
    val sas = WiringInfo("A", "r[a]", sinks, "pin", "A")
    val input =
      """circuit Top :
        |  module Top :
        |    input clock: Clock
        |    inst a of A
        |    a.clock <= clock
        |  module A :
        |    input clock: Clock
        |    reg r: UInt<2>[5], clock
        |    node a = UInt(5)
        |    inst x of X
        |    x.clock <= clock
        |  extmodule X :
        |    input clock: Clock
        |""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    input clock: Clock
        |    inst a of A
        |    a.clock <= clock
        |  module A :
        |    input clock: Clock
        |    reg r: UInt<2>[5], clock
        |    node a = UInt(5)
        |    inst x of X
        |    x.clock <= clock
        |    wire r_a: UInt<2>
        |    r_a <= r[a]
        |    x.pin <= r_a
        |  extmodule X :
        |    input clock: Clock
        |    input pin: UInt<2>
        |""".stripMargin
    val c = passes.foldLeft(parse(input)) {
      (c: Circuit, p: Pass) => p.run(c)
    }
    val wiringPass = new Wiring(Seq(sas))
    val retC = wiringPass.run(c)
    (parse(retC.serialize).serialize) should be (parse(check).serialize)
  }
}
