// SPDX-License-Identifier: Apache-2.0

package firrtlTests

import firrtl._
import firrtl.passes._
import firrtl.testutils._

class ZeroWidthTests extends FirrtlFlatSpec {
  def transforms = Seq(ToWorkingIR, ResolveKinds, InferTypes, ResolveFlows, new InferWidths, ZeroWidth)
  private def exec(input: String) = {
    val circuit = parse(input)
    transforms
      .foldLeft(CircuitState(circuit, UnknownForm)) { (c: CircuitState, p: Transform) =>
        p.runTransform(c)
      }
      .circuit
      .serialize
  }
  // =============================
  "Zero width port" should " be deleted" in {
    val input =
      """circuit Top :
        |  module Top :
        |    input y : UInt<0>
        |    output x : UInt<1>
        |    x <= y""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    output x : UInt<1>
        |    x <= UInt<1>(0)""".stripMargin
    (parse(exec(input))) should be(parse(check))
  }
  "Add of <0> and <2> " should " put in zero" in {
    val input =
      """circuit Top :
        |  module Top :
        |    input y : UInt<0>
        |    output x : UInt
        |    x <= add(y, UInt<2>(2))""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    output x : UInt<3>
        |    x <= add(UInt<1>(0), UInt<2>(2))""".stripMargin
    (parse(exec(input)).serialize) should be(parse(check).serialize)
  }
  "Mux on <0>" should "put in zero" in {
    val input =
      """circuit Top :
        |  module Top :
        |    input y : UInt<0>
        |    output x : UInt
        |    x <= mux(y, UInt<2>(2), UInt<2>(1))""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    output x : UInt<2>
        |    x <= mux(UInt<1>(0), UInt<2>(2), UInt<2>(1))""".stripMargin
    (parse(exec(input)).serialize) should be(parse(check).serialize)
  }
  "Bundle with field of <0>" should "get deleted" in {
    val input =
      """circuit Top :
        |  module Top :
        |    input y : { a: UInt<0> }
        |    output x : { a: UInt<0>, b: UInt<1>}
        |    x.a <= y.a""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    output x : { b: UInt<1> }
        |    skip""".stripMargin
    (parse(exec(input)).serialize) should be(parse(check).serialize)
  }
  "Vector with type of <0>" should "get deleted" in {
    val input =
      """circuit Top :
        |  module Top :
        |    input y : UInt<0>[10]
        |    output x : UInt<0>[10]
        |    x <= y""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    skip""".stripMargin
    (parse(exec(input)).serialize) should be(parse(check).serialize)
  }
  "Node with <0>" should "be removed" in {
    val input =
      """circuit Top :
        |  module Top :
        |    input y: UInt<0>
        |    node x = y""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    skip""".stripMargin
    (parse(exec(input)).serialize) should be(parse(check).serialize)
  }
  "IsInvalid on <0>" should "be deleted" in {
    val input =
      """circuit Top :
        |  module Top :
        |    output y: UInt<0>
        |    y is invalid""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    skip""".stripMargin
    (parse(exec(input)).serialize) should be(parse(check).serialize)
  }
  "Expression in node with type <0>" should "be replaced by UInt<1>(0)" in {
    val input =
      """circuit Top :
        |  module Top :
        |    input x: UInt<1>
        |    input y: UInt<0>
        |    node z = add(x, y)""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    input x: UInt<1>
        |    node z = add(x, UInt<1>(0))""".stripMargin
    (parse(exec(input)).serialize) should be(parse(check).serialize)
  }
  "Expression in cat with type <0>" should "be removed" in {
    val input =
      """circuit Top :
        |  module Top :
        |    input x: UInt<1>
        |    input y: UInt<0>
        |    node z = cat(x, y)""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    input x: UInt<1>
        |    node z = x""".stripMargin
    (parse(exec(input)).serialize) should be(parse(check).serialize)
  }
  "Nested cats with type <0>" should "be removed" in {
    val input =
      """circuit Top :
        |  module Top :
        |    input x: UInt<0>
        |    input y: UInt<0>
        |    input z: UInt<0>
        |    node a = cat(cat(x, y), z)""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    skip""".stripMargin
    (parse(exec(input)).serialize) should be(parse(check).serialize)
  }
  "Nested cats where one has type <0>" should "be unaffected" in {
    val input =
      """circuit Top :
        |  module Top :
        |    input x: UInt<1>
        |    input y: UInt<0>
        |    input z: UInt<1>
        |    node a = cat(cat(x, y), z)""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    input x: UInt<1>
        |    input z: UInt<1>
        |    node a = cat(x, z)""".stripMargin
    (parse(exec(input)).serialize) should be(parse(check).serialize)
  }
  "Stop with type <0>" should "be replaced with UInt(0)" in {
    val input =
      """circuit Top :
        |  module Top :
        |    input clk: Clock
        |    input x: UInt<1>
        |    input y: UInt<0>
        |    input z: UInt<1>
        |    stop(clk, y, 1)""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    input clk: Clock
        |    input x: UInt<1>
        |    input z: UInt<1>
        |    stop(clk, UInt(0), 1)""".stripMargin
    (parse(exec(input)).serialize) should be(parse(check).serialize)
  }
  "Print with type <0>" should "be replaced with UInt(0)" in {
    val input =
      """circuit Top :
        |  module Top :
        |    input clk: Clock
        |    input x: UInt<1>
        |    input y: UInt<0>
        |    input z: UInt<1>
        |    printf(clk, UInt(1), "%d %d %d\n", x, y, z)""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    input clk: Clock
        |    input x: UInt<1>
        |    input z: UInt<1>
        |    printf(clk, UInt(1), "%d %d %d\n", x, UInt(0), z)""".stripMargin
    (parse(exec(input)).serialize) should be(parse(check).serialize)
  }

  "Andr of zero-width expression" should "return true" in {
    val input =
      """circuit Top :
        |  module Top :
        |    input y : UInt<0>
        |    output x : UInt<1>
        |    x <= andr(y)""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    output x : UInt<1>
        |    x <= UInt<1>(1)""".stripMargin
    (parse(exec(input))) should be(parse(check))
  }

  "Cat of SInt with zero-width" should "keep type correctly" in {
    val input =
      """circuit Top :
        |  module Top :
        |    input x : SInt<0>
        |    input y : SInt<1>
        |    output z : UInt<1>
        |    z <= cat(y, x)""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    input y : SInt<1>
        |    output z : UInt<1>
        |    z <= asUInt(y)""".stripMargin
    (parse(exec(input))) should be(parse(check))
  }

  "Memories with zero-width data-type" should "be fully removed" in {
    val input =
      """circuit Foo:
        |  module Foo:
        |    input clock: Clock
        |    input rAddr: UInt<4>
        |    input rEn: UInt<1>
        |    output rData: UInt<0>
        |    input wAddr: UInt<4>
        |    input wEn: UInt<1>
        |    input wMask: UInt<1>
        |    input wData: UInt<0>
        |    input rwEn: UInt<1>
        |    input rwMode: UInt<1>
        |    input rwAddr: UInt<1>
        |    input rwMask: UInt<1>
        |    input rwDataIn: UInt<0>
        |    output rwDataOut: UInt<0>
        |
        |    mem memory:
        |      data-type => UInt<0>
        |      depth => 16
        |      reader => r
        |      writer => w
        |      readwriter => rw
        |      read-latency => 0
        |      write-latency => 1
        |      read-under-write => undefined
        |
        |    memory.r.clk <= clock
        |    memory.r.en <= rEn
        |    memory.r.addr <= rAddr
        |    rData <= memory.r.data
        |    memory.w.clk <= clock
        |    memory.w.en <= wEn
        |    memory.w.addr <= wAddr
        |    memory.w.mask <= wMask
        |    memory.w.data <= wData
        |    memory.rw.clk <= clock
        |    memory.rw.en <= rwEn
        |    memory.rw.addr <= rwAddr
        |    memory.rw.wmode <= rwMode
        |    memory.rw.wmask <= rwMask
        |    memory.rw.wdata <= rwDataIn
        |    rwDataOut <= memory.rw.rdata""".stripMargin
    val check =
      s"""circuit Foo:
         |  module Foo:
         |    input clock: Clock
         |    input rAddr: UInt<4>
         |    input rEn: UInt<1>
         |    input wAddr: UInt<4>
         |    input wEn: UInt<1>
         |    input wMask: UInt<1>
         |    input rwEn: UInt<1>
         |    input rwMode: UInt<1>
         |    input rwAddr: UInt<1>
         |    input rwMask: UInt<1>
         |
         |${Seq.tabulate(17)(_ => "    skip").mkString("\n")}""".stripMargin
    parse(exec(input)) should be(parse(check))
  }
}

class ZeroWidthVerilog extends FirrtlFlatSpec {
  "Circuit" should "accept zero width wires" in {
    val compiler = new VerilogCompiler
    val input =
      """circuit Top :
        |  module Top :
        |    input y: UInt<0>
        |    output x: UInt<3>
        |    x <= y""".stripMargin
    val check =
      """module Top(
        |  output  [2:0] x
        |);
        |  assign x = 3'h0;
        |endmodule
        |""".stripMargin.split("\n").map(normalized)
    executeTest(input, check, compiler)
  }
}
