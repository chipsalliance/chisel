package firrtl.backends.experimental.smt.random

import firrtl.options.Dependency
import firrtl.testutils.LeanTransformSpec

class UndefinedMemoryBehaviorSpec extends LeanTransformSpec(Seq(Dependency(UndefinedMemoryBehaviorPass))) {
  behavior.of("UndefinedMemoryBehaviorPass")

  it should "model write-write conflicts between 2 ports" in {

    val circuit = compile(UBMSources.writeWriteConflict, List()).circuit
    // println(circuit.serialize)
    val result = circuit.serialize.split('\n').map(_.trim)

    // a random value should be declared for the data written on a write-write conflict
    assert(result.contains("rand m_a_wwc_data : UInt<32>, m_a_clk when m_a_b_wwc"))

    // a write-write conflict occurs when both ports are enabled and the addresses match
    assert(result.contains("m_a_b_wwc <= and(and(m_a_en, m_b_en), eq(m_a_addr, m_b_addr))"))

    // the data of read port a depends on whether there is a write-write conflict
    assert(result.contains("m.a.data <= mux(m_a_b_wwc, m_a_wwc_data, m_a_data)"))

    // the enable of read port b depends on whether there is a write-write conflict
    assert(result.contains("m.b.en <= and(m_b_en, not(m_a_b_wwc))"))
  }

  it should "model write-write conflicts between 3 ports" in {

    val circuit = compile(UBMSources.writeWriteConflict3, List()).circuit
    //println(circuit.serialize)
    val result = circuit.serialize.split('\n').map(_.trim)

    // when there is more than one next write port, a "active" node is created
    assert(result.contains("node m_a_wwc_active = or(m_a_b_wwc, m_a_c_wwc)"))

    // a random value should be declared for the data written on a write-write conflict
    assert(result.contains("rand m_a_wwc_data : UInt<32>, m_a_clk when m_a_wwc_active"))
    assert(result.contains("rand m_b_wwc_data : UInt<32>, m_b_clk when m_b_c_wwc"))

    // a write-write conflict occurs when both ports are enabled and the addresses match
    Seq(("a", "b"), ("a", "c"), ("b", "c")).foreach {
      case (w1, w2) =>
        assert(
          result.contains(s"m_${w1}_${w2}_wwc <= and(and(m_${w1}_en, m_${w2}_en), eq(m_${w1}_addr, m_${w2}_addr))")
        )
    }

    // the data of read port a depends on whether there is a write-write conflict
    assert(result.contains("m.a.data <= mux(m_a_wwc_active, m_a_wwc_data, m_a_data)"))

    // the data of read port b depends on whether there is a write-write conflict
    assert(result.contains("m.b.data <= mux(m_b_c_wwc, m_b_wwc_data, m_b_data)"))

    // the enable of read port b depends on whether there is a write-write conflict
    assert(result.contains("m.b.en <= and(m_b_en, not(m_a_b_wwc))"))

    // the enable of read port c depends on whether there is a write-write conflict
    // note that in this case we do not add an extra node since the disjunction is only used once
    assert(result.contains("m.c.en <= and(m_c_en, not(or(m_a_c_wwc, m_b_c_wwc)))"))
  }

  it should "model write-write conflicts more efficiently when ports are mutually exclusive" in {

    val circuit = compile(UBMSources.writeWriteConflict3Exclusive, List()).circuit
    // println(circuit.serialize)
    val result = circuit.serialize.split('\n').map(_.trim)

    // we should not compute the conflict between a and c since it is impossible
    assert(!result.contains("node m_a_c_wwc = and(and(m_a_en, m_c_en), eq(m_a_addr, m_c_addr))"))

    // the enable of port b depends on whether there is a conflict with a
    assert(result.contains("m.b.en <= and(m_b_en, not(m_a_b_wwc))"))

    // the data of port b depends on whether these is a conflict with c
    assert(result.contains("m.b.data <= mux(m_b_c_wwc, m_b_wwc_data, m_b_data)"))

    // the enable of port c only depend on whether there is a conflict with b since c and a cannot conflict
    assert(result.contains("m.c.en <= and(m_c_en, not(m_b_c_wwc))"))

    // the data of port a only depends on whether there is a conflict with b since a and c cannot conflict
    assert(result.contains("m.a.data <= mux(m_a_b_wwc, m_a_wwc_data, m_a_data)"))
  }

  it should "assert out-of-bounds writes when told to" in {
    val anno = List(UndefinedMemoryBehaviorOptions(assertNoOutOfBoundsWrites = true))

    val circuit = compile(UBMSources.readWrite(30, 0), anno).circuit
    // println(circuit.serialize)
    val result = circuit.serialize.split('\n').map(_.trim)

    assert(
      result.contains(
        """assert(m_a_clk, geq(UInt<5>("h1e"), m_a_addr), UInt<1>("h1"), "out of bounds read")"""
      )
    )
  }

  it should "model out-of-bounds reads" in {
    val circuit = compile(UBMSources.readWrite(30, 0), List()).circuit
    //println(circuit.serialize)
    val result = circuit.serialize.split('\n').map(_.trim)

    // an out of bounds read happens if the depth is not greater or equal to the address
    assert(result.contains("node m_r_oob = not(geq(UInt<5>(\"h1e\"), m_r_addr))"))

    // the source of randomness needs to be triggered when there is an out of bounds read
    assert(result.contains("rand m_r_rand_data : UInt<32>, m_r_clk when m_r_oob"))

    // the data is random when there is an oob
    assert(result.contains("m_r_data <= mux(m_r_oob, m_r_rand_data, m.r.data)"))
  }

  it should "model un-enabled reads w/o out-of-bounds" in {
    // without possible out-of-bounds
    val circuit = compile(UBMSources.readEnable(32), List()).circuit
    //println(circuit.serialize)
    val result = circuit.serialize.split('\n').map(_.trim)

    // the memory is disabled when it is not enabled
    assert(result.contains("node m_r_disabled = not(m_r_en)"))

    // the source of randomness needs to be triggered when there is an read while the port is disabled
    assert(result.contains("rand m_r_rand_data : UInt<32>, m_r_clk when m_r_disabled"))

    // the data is random when there is an un-enabled read
    assert(result.contains("m_r_data <= mux(m_r_disabled, m_r_rand_data, m.r.data)"))
  }

  it should "model un-enabled reads with out-of-bounds" in {
    // with possible out-of-bounds
    val circuit = compile(UBMSources.readEnable(30), List()).circuit
    //println(circuit.serialize)
    val result = circuit.serialize.split('\n').map(_.trim)

    // the memory is disabled when it is not enabled
    assert(result.contains("node m_r_disabled = not(m_r_en)"))

    // an out of bounds read happens if the depth is not greater or equal to the address and the memory is enabled
    assert(result.contains("node m_r_oob = and(m_r_en, not(geq(UInt<5>(\"h1e\"), m_r_addr)))"))

    // the two possible issues are combined into a single signal
    assert(result.contains("node m_r_do_rand = or(m_r_disabled, m_r_oob)"))

    // the source of randomness needs to be triggered when either issue occurs
    assert(result.contains("rand m_r_rand_data : UInt<32>, m_r_clk when m_r_do_rand"))

    // the data is random when either issue occurs
    assert(result.contains("m_r_data <= mux(m_r_do_rand, m_r_rand_data, m.r.data)"))
  }

  it should "model un-enabled reads with out-of-bounds with read pipelining" in {
    // with read latency one, we need to pipeline the `do_rand` signal
    val circuit = compile(UBMSources.readEnable(30, 1), List()).circuit
    //println(circuit.serialize)
    val result = circuit.serialize.split('\n').map(_.trim)

    // pipeline register
    assert(result.contains("m_r_do_rand_r1 <= m_r_do_rand"))

    // the source of randomness needs to be triggered by the pipeline register
    assert(result.contains("rand m_r_rand_data : UInt<32>, m_r_clk when m_r_do_rand_r1"))

    // the data is random when the pipeline register is 1
    assert(result.contains("m_r_data <= mux(m_r_do_rand_r1, m_r_rand_data, m.r.data)"))
  }

  it should "model read/write conflicts when they are undefined" in {
    val circuit = compile(UBMSources.readWrite(32, 1), List()).circuit
    //println(circuit.serialize)
    val result = circuit.serialize.split('\n').map(_.trim)

    // detect read/write conflicts
    assert(result.contains("m_r_a_rwc <= eq(m_r_addr, m_a_addr)"))

    // delay the signal
    assert(result.contains("m_r_do_rand_r1 <= m_r_rwc"))

    // randomize the data
    assert(result.contains("rand m_r_rand_data : UInt<32>, m_r_clk when m_r_do_rand_r1"))
    assert(result.contains("m_r_data <= mux(m_r_do_rand_r1, m_r_rand_data, m.r.data)"))
  }
}

private object UBMSources {

  val writeWriteConflict =
    s"""
       |circuit Test:
       |  module Test:
       |    input c : Clock
       |    input preset: AsyncReset
       |    input addr : UInt<8>
       |    input data : UInt<32>
       |    input aEn : UInt<1>
       |    input bEn : UInt<1>
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
       |    m.r.clk <= c
       |    m.r.en <= UInt(1)
       |    m.r.addr <= addr
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
       """.stripMargin

  val writeWriteConflict3 =
    s"""
       |circuit Test:
       |  module Test:
       |    input c : Clock
       |    input preset: AsyncReset
       |    input addr : UInt<8>
       |    input data : UInt<32>
       |    input aEn : UInt<1>
       |    input bEn : UInt<1>
       |    input cEn : UInt<1>
       |
       |    mem m:
       |      data-type => UInt<32>
       |      depth => 32
       |      reader => r
       |      writer => a, b, c
       |      read-latency => 0
       |      write-latency => 1
       |      read-under-write => undefined
       |
       |    m.r.clk <= c
       |    m.r.en <= UInt(1)
       |    m.r.addr <= addr
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
       |    m.c.clk <= c
       |    m.c.en <= cEn
       |    m.c.addr <= addr
       |    m.c.data <= data
       |    m.c.mask <= UInt(1)
       """.stripMargin

  val writeWriteConflict3Exclusive =
    s"""
       |circuit Test:
       |  module Test:
       |    input c : Clock
       |    input preset: AsyncReset
       |    input addr : UInt<8>
       |    input data : UInt<32>
       |    input aEn : UInt<1>
       |    input bEn : UInt<1>
       |
       |    mem m:
       |      data-type => UInt<32>
       |      depth => 32
       |      reader => r
       |      writer => a, b, c
       |      read-latency => 0
       |      write-latency => 1
       |      read-under-write => undefined
       |
       |    m.r.clk <= c
       |    m.r.en <= UInt(1)
       |    m.r.addr <= addr
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
       |    m.c.clk <= c
       |    m.c.en <= not(aEn)
       |    m.c.addr <= addr
       |    m.c.data <= data
       |    m.c.mask <= UInt(1)
       """.stripMargin

  def readWrite(depth: Int, readLatency: Int) =
    s"""circuit CollisionTest:
       |  module CollisionTest:
       |    input c : Clock
       |    input preset: AsyncReset
       |    input addr : UInt<8>
       |    input data : UInt<32>
       |    output dataOut : UInt<32>
       |
       |    mem m:
       |      data-type => UInt<32>
       |      depth => $depth
       |      reader => r
       |      writer => a
       |      read-latency => $readLatency
       |      write-latency => 1
       |      read-under-write => undefined
       |
       |    m.r.clk <= c
       |    m.r.en <= UInt(1)
       |    m.r.addr <= addr
       |    dataOut <= m.r.data
       |
       |    m.a.clk <= c
       |    m.a.mask <= UInt(1)
       |    m.a.en <= UInt(1)
       |    m.a.addr <= addr
       |    m.a.data <= data
       |""".stripMargin

  def readEnable(depth: Int, latency: Int = 0) =
    s"""circuit Test:
       |  module Test:
       |    input c : Clock
       |    input addr : UInt<8>
       |    input en : UInt<1>
       |    output data : UInt<32>
       |
       |    mem m:
       |      data-type => UInt<32>
       |      depth => $depth
       |      reader => r
       |      read-latency => $latency
       |      write-latency => 1
       |      read-under-write => old
       |
       |    m.r.clk <= c
       |    m.r.en <= en
       |    m.r.addr <= addr
       |    data <= m.r.data
       |""".stripMargin
}
