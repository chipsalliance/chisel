// See LICENSE for license details.

package firrtlTests

import java.io._
import org.scalatest._
import org.scalatest.prop._
import firrtl._
import firrtl.ir.Circuit
import firrtl.passes._
import firrtl.Parser.IgnoreInfo
import Annotations._
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
    val sinks = Map(("X"-> "pin"))
    val sas = WiringInfo("C", "r", sinks, "A")
    val input =
      """circuit Top :
        |  module Top :
        |    input clk: Clock
        |    inst a of A
        |    a.clk <= clk
        |  module A :
        |    input clk: Clock
        |    inst b of B
        |    b.clk <= clk
        |    inst x of X
        |    x.clk <= clk
        |    inst d of D
        |    d.clk <= clk
        |  module B :
        |    input clk: Clock
        |    inst c of C
        |    c.clk <= clk
        |    inst d of D
        |    d.clk <= clk
        |  module C :
        |    input clk: Clock
        |    reg r: UInt<5>, clk
        |  module D :
        |    input clk: Clock
        |    inst x1 of X
        |    x1.clk <= clk
        |    inst x2 of X
        |    x2.clk <= clk
        |  extmodule X :
        |    input clk: Clock
        |""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    input clk: Clock
        |    inst a of A
        |    a.clk <= clk
        |  module A :
        |    input clk: Clock
        |    inst b of B
        |    b.clk <= clk
        |    inst x of X
        |    x.clk <= clk
        |    inst d of D
        |    d.clk <= clk
        |    wire r: UInt<5>
        |    r <= b.r
        |    x.pin <= r
        |    d.r <= r
        |  module B :
        |    input clk: Clock
        |    output r: UInt<5>
        |    inst c of C
        |    c.clk <= clk
        |    inst d of D
        |    d.clk <= clk
        |    r <= c.r_0
        |    d.r <= r
        |  module C :
        |    input clk: Clock
        |    output r_0: UInt<5>
        |    reg r: UInt<5>, clk
        |    r_0 <= r
        |  module D :
        |    input clk: Clock
        |    input r: UInt<5>
        |    inst x1 of X
        |    x1.clk <= clk
        |    inst x2 of X
        |    x2.clk <= clk
        |    x1.pin <= r
        |    x2.pin <= r
        |  extmodule X :
        |    input clk: Clock
        |    input pin: UInt<5>
        |""".stripMargin
    val c = passes.foldLeft(parse(input)) {
      (c: Circuit, p: Pass) => p.run(c)
    }
    val wiringPass = new Wiring(sas)
    val retC = wiringPass.run(c)
    (parse(retC.serialize).serialize) should be (parse(check).serialize)
  }

  "Wiring from r.x to X" should "work" in {
    val sinks = Map(("X"-> "pin"))
    val sas = WiringInfo("A", "r.x", sinks, "A")
    val input =
      """circuit Top :
        |  module Top :
        |    input clk: Clock
        |    inst a of A
        |    a.clk <= clk
        |  module A :
        |    input clk: Clock
        |    reg r : {x: UInt<5>}, clk
        |    inst x of X
        |    x.clk <= clk
        |  extmodule X :
        |    input clk: Clock
        |""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    input clk: Clock
        |    inst a of A
        |    a.clk <= clk
        |  module A :
        |    input clk: Clock
        |    reg r: {x: UInt<5>}, clk
        |    inst x of X
        |    x.clk <= clk
        |    wire r_x: UInt<5>
        |    r_x <= r.x
        |    x.pin <= r_x
        |  extmodule X :
        |    input clk: Clock
        |    input pin: UInt<5>
        |""".stripMargin
    val c = passes.foldLeft(parse(input)) {
      (c: Circuit, p: Pass) => p.run(c)
    }
    val wiringPass = new Wiring(sas)
    val retC = wiringPass.run(c)
    (parse(retC.serialize).serialize) should be (parse(check).serialize)
  }
  "Wiring from clk to X" should "work" in {
    val sinks = Map(("X"-> "pin"))
    val sas = WiringInfo("A", "clk", sinks, "A")
    val input =
      """circuit Top :
        |  module Top :
        |    input clk: Clock
        |    inst a of A
        |    a.clk <= clk
        |  module A :
        |    input clk: Clock
        |    inst x of X
        |    x.clk <= clk
        |  extmodule X :
        |    input clk: Clock
        |""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    input clk: Clock
        |    inst a of A
        |    a.clk <= clk
        |  module A :
        |    input clk: Clock
        |    inst x of X
        |    x.clk <= clk
        |    wire clk_0: Clock
        |    clk_0 <= clk
        |    x.pin <= clk_0
        |  extmodule X :
        |    input clk: Clock
        |    input pin: Clock
        |""".stripMargin
    val c = passes.foldLeft(parse(input)) {
      (c: Circuit, p: Pass) => p.run(c)
    }
    val wiringPass = new Wiring(sas)
    val retC = wiringPass.run(c)
    (parse(retC.serialize).serialize) should be (parse(check).serialize)
  }
  "Two sources" should "fail" in {
    val sinks = Map(("X"-> "pin"))
    val sas = WiringInfo("A", "clk", sinks, "Top")
    val input =
      """circuit Top :
        |  module Top :
        |    input clk: Clock
        |    inst a1 of A
        |    a1.clk <= clk
        |    inst a2 of A
        |    a2.clk <= clk
        |  module A :
        |    input clk: Clock
        |    inst x of X
        |    x.clk <= clk
        |  extmodule X :
        |    input clk: Clock
        |""".stripMargin
    intercept[WiringException] {
      val c = passes.foldLeft(parse(input)) {
        (c: Circuit, p: Pass) => p.run(c)
      }
      val wiringPass = new Wiring(sas)
      val retC = wiringPass.run(c)
    }
  }
  "Wiring from A.clk to X, with 2 A's, and A as top" should "work" in {
    val sinks = Map(("X"-> "pin"))
    val sas = WiringInfo("A", "clk", sinks, "A")
    val input =
      """circuit Top :
        |  module Top :
        |    input clk: Clock
        |    inst a1 of A
        |    a1.clk <= clk
        |    inst a2 of A
        |    a2.clk <= clk
        |  module A :
        |    input clk: Clock
        |    inst x of X
        |    x.clk <= clk
        |  extmodule X :
        |    input clk: Clock
        |""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    input clk: Clock
        |    inst a1 of A
        |    a1.clk <= clk
        |    inst a2 of A
        |    a2.clk <= clk
        |  module A :
        |    input clk: Clock
        |    inst x of X
        |    x.clk <= clk
        |    wire clk_0: Clock
        |    clk_0 <= clk
        |    x.pin <= clk_0
        |  extmodule X :
        |    input clk: Clock
        |    input pin: Clock
        |""".stripMargin
    val c = passes.foldLeft(parse(input)) {
      (c: Circuit, p: Pass) => p.run(c)
    }
    val wiringPass = new Wiring(sas)
    val retC = wiringPass.run(c)
    (parse(retC.serialize).serialize) should be (parse(check).serialize)
  }
  "Wiring from A.clk to X, with 2 A's, and A as top, but Top instantiates X" should "error" in {
    val sinks = Map(("X"-> "pin"))
    val sas = WiringInfo("A", "clk", sinks, "A")
    val input =
      """circuit Top :
        |  module Top :
        |    input clk: Clock
        |    inst a1 of A
        |    a1.clk <= clk
        |    inst a2 of A
        |    a2.clk <= clk
        |    inst x of X
        |    x.clk <= clk
        |  module A :
        |    input clk: Clock
        |    inst x of X
        |    x.clk <= clk
        |  extmodule X :
        |    input clk: Clock
        |""".stripMargin
    intercept[WiringException] {
      val c = passes.foldLeft(parse(input)) {
        (c: Circuit, p: Pass) => p.run(c)
      }
      val wiringPass = new Wiring(sas)
      val retC = wiringPass.run(c)
    }
  }
  "Wiring from A.r[a] to X" should "work" in {
    val sinks = Map(("X"-> "pin"))
    val sas = WiringInfo("A", "r[a]", sinks, "A")
    val input =
      """circuit Top :
        |  module Top :
        |    input clk: Clock
        |    inst a of A
        |    a.clk <= clk
        |  module A :
        |    input clk: Clock
        |    reg r: UInt<2>[5], clk
        |    node a = UInt(5)
        |    inst x of X
        |    x.clk <= clk
        |  extmodule X :
        |    input clk: Clock
        |""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    input clk: Clock
        |    inst a of A
        |    a.clk <= clk
        |  module A :
        |    input clk: Clock
        |    reg r: UInt<2>[5], clk
        |    node a = UInt(5)
        |    inst x of X
        |    x.clk <= clk
        |    wire r_a: UInt<2>
        |    r_a <= r[a]
        |    x.pin <= r_a
        |  extmodule X :
        |    input clk: Clock
        |    input pin: UInt<2>
        |""".stripMargin
    val c = passes.foldLeft(parse(input)) {
      (c: Circuit, p: Pass) => p.run(c)
    }
    val wiringPass = new Wiring(sas)
    val retC = wiringPass.run(c)
    (parse(retC.serialize).serialize) should be (parse(check).serialize)
  }
}
