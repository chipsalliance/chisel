// SPDX-License-Identifier: Apache-2.0

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

  "Registered read-first memory" should "return written data after two cycles" taggedAs (RequiresZ3) in {
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

  "Registered read-first memory" should "not return written data after one cycle" taggedAs (RequiresZ3) in {
    val cmds =
      """
        |reg past_in: UInt<8>, clock
        |past_in <= in
        |
        |assume(clock, eq(read_addr, write_addr), UInt(1), "read_addr = write_addr")
        |assert(clock, eq(out, past_in), past_valid, "out = past(in)")
        |""".stripMargin
    test(registeredTestMem("Mem00", cmds, "old"), MCFail(1), kmax = 3)
  }

  "Registered write-first memory" should "return written data after one cycle" taggedAs (RequiresZ3) in {
    val cmds =
      """
        |reg past_in: UInt<8>, clock
        |past_in <= in
        |
        |assume(clock, eq(read_addr, write_addr), UInt(1), "read_addr = write_addr")
        |assert(clock, eq(out, past_in), past_valid, "out = past(in)")
        |""".stripMargin
    test(registeredTestMem("Mem00", cmds, "new"), MCSuccess, kmax = 3)
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

  def collisionTest(assumption: String) = s"""
                                             |circuit CollisionTest:
                                             |  module CollisionTest:
                                             |    input c : Clock
                                             |    input preset: AsyncReset
                                             |    input addr : UInt<8>
                                             |    input data : UInt<32>
                                             |    input aEn : UInt<1>
                                             |    input bEn : UInt<1>
                                             |
                                             |    reg cycle: UInt<8>, c with: (reset => (preset, UInt(0)))
                                             |    cycle <= add(cycle, UInt(1))
                                             |    node pastValid = geq(cycle, UInt(1))
                                             |
                                             |    reg prevAddr: UInt<8>, c
                                             |    prevAddr <= addr
                                             |    reg prevData: UInt<32>, c
                                             |    prevData <= data
                                             |    reg prevEn: UInt<1>, c
                                             |    prevEn <= or(aEn, bEn)
                                             |
                                             |    mem m:
                                             |      data-type => UInt<32>
                                             |      depth => 32
                                             |      reader => r
                                             |      writer => a, b
                                             |      read-latency => 0
                                             |      write-latency => 1
                                             |      read-under-write => undefined
                                             |
                                             |    ; the readport is used to verify the written value
                                             |    m.r.clk <= c
                                             |    m.r.en <= UInt(1)
                                             |    m.r.addr <= prevAddr
                                             |
                                             |    ; both read ports write to the same address and the same data
                                             |    m.a.clk <= c
                                             |    m.a.en <= aEn
                                             |    m.a.addr <= addr
                                             |    m.a.data <= data
                                             |    m.a.mask <= UInt(1)
                                             |    m.b.clk <= c
                                             |    m.b.en <= bEn
                                             |    m.b.addr <= addr
                                             |    m.b.data <= data
                                             |    m.b.mask <= UInt(1)
                                             |
                                             |    ; we assume that writes are mutually exclusive
                                             |    ; => no collision should occur
                                             |    assume(c, $assumption, UInt(1), "")
                                             |
                                             |    ; we check that we always read the last written value
                                             |    assert(c, eq(m.r.data, prevData), and(pastValid, prevEn), "")
                                             |""".stripMargin
  "memory with two write ports" should "not have collisions when enables are mutually exclusive" taggedAs (RequiresZ3) in {
    test(collisionTest("not(and(aEn, bEn))"), MCSuccess, kmax = 4)
  }
  "memory with two write ports" should "can have collisions when enables are unconstrained" taggedAs (RequiresZ3) in {
    test(collisionTest("UInt(1)"), MCFail(1), kmax = 1)
  }

  private def readEnableSrc(pred: String, num: Int) =
    s"""
       |circuit ReadEnableTest$num:
       |  module ReadEnableTest$num:
       |    input c : Clock
       |    input preset: AsyncReset
       |
       |    reg first: UInt<1>, c with: (reset => (preset, UInt(1)))
       |    first <= UInt(0)
       |
       |    reg even: UInt<1>, c with: (reset => (preset, UInt(0)))
       |    node odd = not(even)
       |    even <= not(even)
       |
       |    mem m:
       |      data-type => UInt<8>
       |      depth => 4
       |      reader => r
       |      read-latency => 1
       |      write-latency => 1
       |      read-under-write => undefined
       |
       |    m.r.clk <= c
       |    m.r.addr <= UInt(0)
       |    ; the read port is enabled in even cycles
       |    m.r.en <= even
       |
       |    assert(c, $pred, not(first), "")
       |""".stripMargin

  "a memory with read enable" should "supply valid data one cycle after en=1" in {
    val init = Seq(MemoryScalarInitAnnotation(CircuitTarget(s"ReadEnableTest1").module(s"ReadEnableTest1").ref("m"), 0))
    // the read port is enabled on even cycles, so on odd cycles we should reliably get zeros
    test(readEnableSrc("or(not(odd), eq(m.r.data, UInt(0)))", 1), MCSuccess, kmax = 3, annos = init)
  }

  "a memory with read enable" should "supply invalid data one cycle after en=0" in {
    val init = Seq(MemoryScalarInitAnnotation(CircuitTarget(s"ReadEnableTest2").module(s"ReadEnableTest2").ref("m"), 0))
    // the read port is disabled on odd cycles, so on even cycles we should *NOT* reliably get zeros
    test(readEnableSrc("or(not(even), eq(m.r.data, UInt(0)))", 2), MCFail(1), kmax = 1, annos = init)
  }
}
