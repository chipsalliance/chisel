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

  it should "wire from a register source (r) to multiple extmodule sinks (X)" in {
    val sinks = Seq(ModuleName("X", CircuitName("Top")))
    val source = ComponentName("r", ModuleName("C", CircuitName("Top")))
    val sas = WiringInfo(source, sinks, "pin")
    val input =
      """|circuit Top :
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
      """|circuit Top :
         |  module Top :
         |    input clock: Clock
         |    inst a of A
         |    a.clock <= clock
         |  module A :
         |    input clock: Clock
         |    wire r: UInt<5>
         |    inst b of B
         |    b.clock <= clock
         |    inst x of X
         |    x.clock <= clock
         |    inst d of D
         |    d.clock <= clock
         |    d.r <= r
         |    r <= b.r
         |    x.pin <= r
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

  it should "wire from a register source (r) to multiple module sinks (X)" in {
    val sinks = Seq(ModuleName("X", CircuitName("Top")))
    val source = ComponentName("r", ModuleName("C", CircuitName("Top")))
    val sas = WiringInfo(source, sinks, "pin")
    val input =
      """|circuit Top :
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
         |  module X :
         |    input clock: Clock
         |""".stripMargin
    val check =
      """|circuit Top :
         |  module Top :
         |    input clock: Clock
         |    inst a of A
         |    a.clock <= clock
         |  module A :
         |    input clock: Clock
         |    wire r: UInt<5>
         |    inst b of B
         |    b.clock <= clock
         |    inst x of X
         |    x.clock <= clock
         |    inst d of D
         |    d.clock <= clock
         |    d.r <= r
         |    r <= b.r
         |    x.pin <= r
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
         |  module X :
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

  it should "wire from a register sink (r) to a wire source (s) in another module (X)" in {
    val sinks = Seq(ComponentName("s", ModuleName("X", CircuitName("Top"))))
    val source = ComponentName("r", ModuleName("C", CircuitName("Top")))
    val sas = WiringInfo(source, sinks, "pin")
    val input =
      """|circuit Top :
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
         |  module X :
         |    input clock: Clock
         |    wire s: UInt<5>
         |""".stripMargin
    val check =
      """|circuit Top :
         |  module Top :
         |    input clock: Clock
         |    inst a of A
         |    a.clock <= clock
         |  module A :
         |    input clock: Clock
         |    wire r: UInt<5>
         |    inst b of B
         |    b.clock <= clock
         |    inst x of X
         |    x.clock <= clock
         |    inst d of D
         |    d.clock <= clock
         |    d.r <= r
         |    r <= b.r
         |    x.pin <= r
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
         |  module X :
         |    input clock: Clock
         |    input pin: UInt<5>
         |    wire s: UInt<5>
         |    s <= pin
         |""".stripMargin
    val c = passes.foldLeft(parse(input)) {
      (c: Circuit, p: Pass) => p.run(c)
    }
    val wiringPass = new Wiring(Seq(sas))
    val retC = wiringPass.run(c)
    (parse(retC.serialize).serialize) should be (parse(check).serialize)
  }

  it should "wire from a SubField source (r.x) to an extmodule sink (X)" in {
    val sinks = Seq(ModuleName("X", CircuitName("Top")))
    val source = ComponentName("r.x", ModuleName("A", CircuitName("Top")))
    val sas = WiringInfo(source, sinks, "pin")
    val input =
      """|circuit Top :
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
      """|circuit Top :
         |  module Top :
         |    input clock: Clock
         |    inst a of A
         |    a.clock <= clock
         |  module A :
         |    input clock: Clock
         |    wire r_x: UInt<5>
         |    reg r: {x: UInt<5>}, clock
         |    inst x of X
         |    x.clock <= clock
         |    x.pin <= r_x
         |    r_x <= r.x
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

  it should "wire properly with a source as a submodule of a sink" in {
    val sinks = Seq(ComponentName("s", ModuleName("A", CircuitName("Top"))))
    val source = ComponentName("r", ModuleName("X", CircuitName("Top")))
    val sas = WiringInfo(source, sinks, "pin")
    val input =
      """|circuit Top :
         |  module Top :
         |    input clock: Clock
         |    inst a of A
         |    a.clock <= clock
         |  module A :
         |    input clock: Clock
         |    wire s: UInt<5>
         |    inst x of X
         |    x.clock <= clock
         |  module X :
         |    input clock: Clock
         |    reg r: UInt<5>, clock
         |""".stripMargin
    val check =
      """|circuit Top :
         |  module Top :
         |    input clock: Clock
         |    inst a of A
         |    a.clock <= clock
         |  module A :
         |    input clock: Clock
         |    wire pin: UInt<5>
         |    wire s: UInt<5>
         |    inst x of X
         |    x.clock <= clock
         |    pin <= x.r_0
         |    s <= pin
         |  module X :
         |    input clock: Clock
         |    output r_0: UInt<5>
         |    reg r: UInt<5>, clock
         |    r_0 <= r
         |""".stripMargin
    val c = passes.foldLeft(parse(input)) {
      (c: Circuit, p: Pass) => p.run(c)
    }
    val wiringPass = new Wiring(Seq(sas))
    val retC = wiringPass.run(c)
    (parse(retC.serialize).serialize) should be (parse(check).serialize)
  }

  it should "wire with source and sink in the same module" in {
    val sinks = Seq(ComponentName("s", ModuleName("Top", CircuitName("Top"))))
    val source = ComponentName("r", ModuleName("Top", CircuitName("Top")))
    val sas = WiringInfo(source, sinks, "pin")
    val input =
      """|circuit Top :
         |  module Top :
         |    input clock: Clock
         |    wire s: UInt<5>
         |    reg r: UInt<5>, clock
         |""".stripMargin
    val check =
      """|circuit Top :
         |  module Top :
         |    input clock: Clock
         |    wire pin: UInt<5>
         |    wire s: UInt<5>
         |    reg r: UInt<5>, clock
         |    s <= pin
         |    pin <= r
         |""".stripMargin
    val c = passes.foldLeft(parse(input)) {
      (c: Circuit, p: Pass) => p.run(c)
    }
    val wiringPass = new Wiring(Seq(sas))
    val retC = wiringPass.run(c)
    (parse(retC.serialize).serialize) should be (parse(check).serialize)
  }

  it should "wire multiple sinks in the same module" in {
    val sinks = Seq(ComponentName("s", ModuleName("A", CircuitName("Top"))),
                    ComponentName("t", ModuleName("A", CircuitName("Top"))))
    val source = ComponentName("r", ModuleName("A", CircuitName("Top")))
    val sas = WiringInfo(source, sinks, "pin")
    val input =
      """|circuit Top :
         |  module Top :
         |    input clock: Clock
         |    inst a of A
         |    a.clock <= clock
         |  module A :
         |    input clock: Clock
         |    wire s: UInt<5>
         |    wire t: UInt<5>
         |    reg r: UInt<5>, clock
         |""".stripMargin
    val check =
      """|circuit Top :
         |  module Top :
         |    input clock: Clock
         |    inst a of A
         |    a.clock <= clock
         |  module A :
         |    input clock: Clock
         |    wire pin: UInt<5>
         |    wire s: UInt<5>
         |    wire t: UInt<5>
         |    reg r: UInt<5>, clock
         |    t <= pin
         |    s <= pin
         |    pin <= r
         |""".stripMargin
    val c = passes.foldLeft(parse(input)) {
      (c: Circuit, p: Pass) => p.run(c)
    }
    val wiringPass = new Wiring(Seq(sas))
    val retC = wiringPass.run(c)
    (parse(retC.serialize).serialize) should be (parse(check).serialize)
  }

  it should "wire clocks" in {
    val sinks = Seq(ModuleName("X", CircuitName("Top")))
    val source = ComponentName("clock", ModuleName("A", CircuitName("Top")))
    val sas = WiringInfo(source, sinks, "pin")
    val input =
      """|circuit Top :
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
      """|circuit Top :
         |  module Top :
         |    input clock: Clock
         |    inst a of A
         |    a.clock <= clock
         |  module A :
         |    input clock: Clock
         |    wire clock_0: Clock
         |    inst x of X
         |    x.clock <= clock
         |    x.pin <= clock_0
         |    clock_0 <= clock
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

  it should "handle two source instances with clearly defined sinks" in {
    val sinks = Seq(ModuleName("X", CircuitName("Top")))
    val source = ComponentName("clock", ModuleName("A", CircuitName("Top")))
    val sas = WiringInfo(source, sinks, "pin")
    val input =
      """|circuit Top :
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
      """|circuit Top :
         |  module Top :
         |    input clock: Clock
         |    inst a1 of A
         |    a1.clock <= clock
         |    inst a2 of A
         |    a2.clock <= clock
         |  module A :
         |    input clock: Clock
         |    wire clock_0: Clock
         |    inst x of X
         |    x.clock <= clock
         |    x.pin <= clock_0
         |    clock_0 <= clock
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

  it should "wire multiple clocks" in {
    val sinks = Seq(ModuleName("X", CircuitName("Top")))
    val source = ComponentName("clock", ModuleName("A", CircuitName("Top")))
    val sas = WiringInfo(source, sinks, "pin")
    val input =
      """|circuit Top :
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
      """|circuit Top :
         |  module Top :
         |    input clock: Clock
         |    inst a1 of A
         |    a1.clock <= clock
         |    inst a2 of A
         |    a2.clock <= clock
         |  module A :
         |    input clock: Clock
         |    wire clock_0: Clock
         |    inst x of X
         |    x.clock <= clock
         |    x.pin <= clock_0
         |    clock_0 <= clock
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

  it should "error with WiringException for indeterminate ownership" in {
    val sinks = Seq(ModuleName("X", CircuitName("Top")))
    val source = ComponentName("clock", ModuleName("A", CircuitName("Top")))
    val sas = WiringInfo(source, sinks, "pin")
    val input =
      """|circuit Top :
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

  it should "wire subindex source to sink" in {
    val sinks = Seq(ModuleName("X", CircuitName("Top")))
    val source = ComponentName("r[a]", ModuleName("A", CircuitName("Top")))
    val sas = WiringInfo(source, sinks, "pin")
    val input =
      """|circuit Top :
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
      """|circuit Top :
         |  module Top :
         |    input clock: Clock
         |    inst a of A
         |    a.clock <= clock
         |  module A :
         |    input clock: Clock
         |    wire r_a: UInt<2>
         |    reg r: UInt<2>[5], clock
         |    node a = UInt(5)
         |    inst x of X
         |    x.clock <= clock
         |    x.pin <= r_a
         |    r_a <= r[a]
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

  it should "wire using Annotations with a sink module" in {
    val source = SourceAnnotation(ComponentName("r", ModuleName("Top", CircuitName("Top"))), "pin")
    val sink = SinkAnnotation(ModuleName("X", CircuitName("Top")), "pin")
    val input =
      """|circuit Top :
         |  module Top :
         |    input clk: Clock
         |    inst x of X
         |    x.clk <= clk
         |    reg r: UInt<5>, clk
         |  extmodule X :
         |    input clk: Clock
         |""".stripMargin
    val check =
      """|circuit Top :
         |  module Top :
         |    input clk: Clock
         |    wire r_0 : UInt<5>
         |    inst x of X
         |    x.clk <= clk
         |    reg r: UInt<5>, clk
         |    x.pin <= r_0
         |    r_0 <= r
         |  extmodule X :
         |    input clk: Clock
         |    input pin: UInt<5>
         |""".stripMargin
    val c = passes.foldLeft(parse(input)) {
      (c: Circuit, p: Pass) => p.run(c)
    }
    val wiringXForm = new WiringTransform()
    val retC = wiringXForm.execute(CircuitState(c, MidForm, Seq(source, sink))).circuit
    (parse(retC.serialize).serialize) should be (parse(check).serialize)
  }

  it should "wire using Annotations with a sink component" in {
    val source = SourceAnnotation(ComponentName("r", ModuleName("Top", CircuitName("Top"))), "pin")
    val sink = SinkAnnotation(ComponentName("s", ModuleName("X", CircuitName("Top"))), "pin")
    val input =
      """|circuit Top :
         |  module Top :
         |    input clk: Clock
         |    inst x of X
         |    x.clk <= clk
         |    reg r: UInt<5>, clk
         |  module X :
         |    input clk: Clock
         |    wire s: UInt<5>
         |""".stripMargin
    val check =
      """|circuit Top :
         |  module Top :
         |    input clk: Clock
         |    wire r_0 : UInt<5>
         |    inst x of X
         |    x.clk <= clk
         |    reg r: UInt<5>, clk
         |    x.pin <= r_0
         |    r_0 <= r
         |  module X :
         |    input clk: Clock
         |    input pin: UInt<5>
         |    wire s: UInt<5>
         |    s <= pin
         |""".stripMargin
    val c = passes.foldLeft(parse(input)) {
      (c: Circuit, p: Pass) => p.run(c)
    }
    val wiringXForm = new WiringTransform()
    val retC = wiringXForm.execute(CircuitState(c, MidForm, Seq(source, sink))).circuit
    (parse(retC.serialize).serialize) should be (parse(check).serialize)
  }

  it should "wire using annotations with Aggregate source" in {
    val source = SourceAnnotation(ComponentName("bundle", ModuleName("A", CircuitName("Top"))), "pin")
    val sink = SinkAnnotation(ModuleName("B", CircuitName("Top")), "pin")
    val input =
      """|circuit Top :
         |  module Top :
         |    input clock : Clock
         |    inst a of A
         |    inst b of B
         |    a.clock <= clock
         |    b.clock <= clock
         |  module A :
         |    input clock : Clock
         |    wire bundle : {x : UInt<1>, y: UInt<1>, z: {zz : UInt<1>} }
         |    bundle is invalid
         |  module B :
         |    input clock : Clock""".stripMargin
    val check =
      """|circuit Top :
         |  module Top :
         |    input clock : Clock
         |    wire bundle : {x : UInt<1>, y: UInt<1>, z: {zz : UInt<1>} }
         |    inst a of A
         |    inst b of B
         |    a.clock <= clock
         |    b.clock <= clock
         |    b.pin <= bundle
         |    bundle <= a.bundle_0
         |  module A :
         |    input clock : Clock
         |    output bundle_0 : {x : UInt<1>, y: UInt<1>, z: {zz : UInt<1>} }
         |    wire bundle : {x : UInt<1>, y: UInt<1>, z: {zz : UInt<1>} }
         |    bundle is invalid
         |    bundle_0 <= bundle
         |  module B :
         |    input clock : Clock
         |    input pin : {x : UInt<1>, y: UInt<1>, z: {zz : UInt<1>} }"""
        .stripMargin
    val c = passes.foldLeft(parse(input)) {
      (c: Circuit, p: Pass) => p.run(c)
    }
    val wiringXForm = new WiringTransform()
    val retC = wiringXForm.execute(CircuitState(c, MidForm, Seq(source, sink))).circuit
    (parse(retC.serialize).serialize) should be (parse(check).serialize)
  }

  it should "wire one sink to multiple, disjoint extmodules" in {
    val sinkX = Seq(ModuleName("X", CircuitName("Top")))
    val sourceX = ComponentName("r.x", ModuleName("A", CircuitName("Top")))
    val sinkY = Seq(ModuleName("Y", CircuitName("Top")))
    val sourceY = ComponentName("r.x", ModuleName("A", CircuitName("Top")))
    val wiSeq = Seq(
      WiringInfo(sourceX, sinkX, "pin"),
      WiringInfo(sourceY, sinkY, "pin"))
    val input =
      """|circuit Top :
         |  module Top :
         |    input clock: Clock
         |    inst a of A
         |    a.clock <= clock
         |  module A :
         |    input clock: Clock
         |    reg r : {x: UInt<5>}, clock
         |    inst x of X
         |    x.clock <= clock
         |    inst y of Y
         |    y.clock <= clock
         |  extmodule X :
         |    input clock: Clock
         |  extmodule Y :
         |    input clock: Clock
         |""".stripMargin
    val check =
      """|circuit Top :
         |  module Top :
         |    input clock: Clock
         |    inst a of A
         |    a.clock <= clock
         |  module A :
         |    input clock: Clock
         |    wire r_x_0: UInt<5>
         |    wire r_x: UInt<5>
         |    reg r: {x: UInt<5>}, clock
         |    inst x of X
         |    x.clock <= clock
         |    inst y of Y
         |    y.clock <= clock
         |    x.pin <= r_x
         |    r_x <= r.x
         |    y.pin <= r_x_0
         |    r_x_0 <= r.x
         |  extmodule X :
         |    input clock: Clock
         |    input pin: UInt<5>
         |  extmodule Y :
         |    input clock: Clock
         |    input pin: UInt<5>
         |""".stripMargin
    val c = passes.foldLeft(parse(input)) {
      (c: Circuit, p: Pass) => p.run(c)
    }
    val wiringPass = new Wiring(wiSeq)
    val retC = wiringPass.run(c)
    (parse(retC.serialize).serialize) should be (parse(check).serialize)
  }
}
