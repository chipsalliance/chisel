// See LICENSE for license details.

package firrtl.backends.experimental.smt.end2end

import firrtl.annotations.{CircuitTarget, MemoryArrayInitAnnotation, MemoryScalarInitAnnotation}

class MemorySpec extends EndToEndSMTBaseSpec {
  private def registeredTestMem(name: String, cmds: String, readUnderWrite: String): String =
    registeredTestMem(name, cmds.split("\n"), readUnderWrite)
  private def registeredTestMem(name: String, cmds: Iterable[String], readUnderWrite: String): String =
    s"""circuit $name:
       |  module $name:
       |    input reset : UInt<1>
       |    input clock : Clock
       |    input preset: AsyncReset
       |    input write_addr : UInt<5>
       |    input read_addr : UInt<5>
       |    input in : UInt<8>
       |    output out : UInt<8>
       |
       |    mem m:
       |      data-type => UInt<8>
       |      depth => 32
       |      reader => r
       |      writer => w
       |      read-latency => 1
       |      write-latency => 1
       |      read-under-write => $readUnderWrite
       |
       |    m.w.clk <= clock
       |    m.w.mask <= UInt(1)
       |    m.w.en <= UInt(1)
       |    m.w.data <= in
       |    m.w.addr <= write_addr
       |
       |    m.r.clk <= clock
       |    m.r.en <= UInt(1)
       |    out <= m.r.data
       |    m.r.addr <= read_addr
       |
       |    reg cycle: UInt<8>, clock with: (reset => (preset, UInt(0)))
       |    cycle <= add(cycle, UInt(1))
       |    node past_valid = geq(cycle, UInt(1))
       |
       |    ${cmds.mkString("\n    ")}
       |""".stripMargin

  "Registered test memory" should "return written data after two cycles" taggedAs (RequiresZ3) in {
    val cmds =
      """node past_past_valid = geq(cycle, UInt(2))
        |reg past_in: UInt<8>, clock
        |past_in <= in
        |reg past_past_in: UInt<8>, clock
        |past_past_in <= past_in
        |reg past_write_addr: UInt<5>, clock
        |past_write_addr <= write_addr
        |
        |assume(clock, eq(read_addr, past_write_addr), past_valid, "read_addr = past(write_addr)")
        |assert(clock, eq(out, past_past_in), past_past_valid, "out = past(past(in))")
        |""".stripMargin
    test(registeredTestMem("Mem00", cmds, "old"), MCSuccess, kmax = 3)
  }

  private def readOnlyMem(pred: String, num: Int) =
    s"""circuit Mem0$num:
       |  module Mem0$num:
       |    input c : Clock
       |    input read_addr : UInt<2>
       |    output out : UInt<8>
       |
       |    mem m:
       |      data-type => UInt<8>
       |      depth => 4
       |      reader => r
       |      read-latency => 0
       |      write-latency => 1
       |      read-under-write => new
       |
       |    m.r.clk <= c
       |    m.r.en <= UInt(1)
       |    out <= m.r.data
       |    m.r.addr <= read_addr
       |
       |    assert(c, $pred, UInt(1), "")
       |""".stripMargin
  private def m(num: Int) = CircuitTarget(s"Mem0$num").module(s"Mem0$num").ref("m")

  "read-only memory" should "always return 0" taggedAs (RequiresZ3) in {
    test(readOnlyMem("eq(out, UInt(0))", 1), MCSuccess, kmax = 2, annos = Seq(MemoryScalarInitAnnotation(m(1), 0)))
  }

  "read-only memory" should "not always return 1" taggedAs (RequiresZ3) in {
    test(readOnlyMem("eq(out, UInt(1))", 2), MCFail(0), kmax = 2, annos = Seq(MemoryScalarInitAnnotation(m(2), 0)))
  }

  "read-only memory" should "always return 1 or 2" taggedAs (RequiresZ3) in {
    test(
      readOnlyMem("or(eq(out, UInt(1)), eq(out, UInt(2)))", 3),
      MCSuccess,
      kmax = 2,
      annos = Seq(MemoryArrayInitAnnotation(m(3), Seq(1, 2, 2, 1)))
    )
  }

  "read-only memory" should "not always return 1 or 2 or 3" taggedAs (RequiresZ3) in {
    test(
      readOnlyMem("or(eq(out, UInt(1)), eq(out, UInt(2)))", 4),
      MCFail(0),
      kmax = 2,
      annos = Seq(MemoryArrayInitAnnotation(m(4), Seq(1, 2, 2, 3)))
    )
  }
}
