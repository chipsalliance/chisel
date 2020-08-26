// See LICENSE for license details.

package firrtl.backends.experimental.smt

import firrtl.{MemoryArrayInit, MemoryScalarInit, Utils}

private class FirrtlModuleToTransitionSystemSpec extends SMTBackendBaseSpec {
  behavior.of("ModuleToTransitionSystem.run")

  it should "model registers as state" in {
    // if a signal is invalid, it could take on an arbitrary value in that cycle
    val src =
      """circuit m:
        |  module m:
        |    input reset : UInt<1>
        |    input clock : Clock
        |    input en : UInt<1>
        |    input in : UInt<8>
        |    output out : UInt<8>
        |
        |    reg r : UInt<8>, clock with : (reset => (reset, UInt<8>(0)))
        |    when en:
        |      r <= in
        |    out <= r
        |
        |""".stripMargin
    val sys = toSys(src)

    assert(sys.signals.length == 2)

    // the when is translated as a ITE
    val genSignal = sys.signals.filterNot(_.name == "out").head
    assert(genSignal.e.toString == "ite(en, in, r)")

    // the reset is synchronous
    val r = sys.states.head
    assert(r.sym.name == "r")
    assert(r.init.isEmpty, "we are not using any preset, so the initial register content is arbitrary")
    assert(r.next.get.toString == s"ite(reset, 8'b0, ${genSignal.name})")
  }

  private def memCircuit(depth: Int = 32) =
    s"""circuit m:
       |  module m:
       |    input reset : UInt<1>
       |    input clock : Clock
       |    input addr : UInt<${Utils.getUIntWidth(depth)}>
       |    input in : UInt<8>
       |    output out : UInt<8>
       |
       |    mem m:
       |      data-type => UInt<8>
       |      depth => $depth
       |      reader => r
       |      writer => w
       |      read-latency => 0
       |      write-latency => 1
       |      read-under-write => new
       |
       |    m.w.clk <= clock
       |    m.w.mask <= UInt(1)
       |    m.w.en <= UInt(1)
       |    m.w.data <= in
       |    m.w.addr <= addr
       |
       |    m.r.clk <= clock
       |    m.r.en <= UInt(1)
       |    out <= m.r.data
       |    m.r.addr <= addr
       |
       |""".stripMargin

  it should "model memories as state" in {
    val sys = toSys(memCircuit())

    assert(sys.signals.length == 9 - 2 + 1, "9 connects - 2 clock connects + 1 combinatorial read port")

    val sig = sys.signals.map(s => s.name -> s.e).toMap

    // masks and enables should all be true
    val True = BVLiteral(1, 1)
    assert(sig("m.w.mask") == True)
    assert(sig("m.w.en") == True)
    assert(sig("m.r.en") == True)

    // read data should always be enabled
    assert(sig("m.r.data").toString == "m[m.r.addr]")

    // the memory is modelled as a state
    val m = sys.states.find(_.sym.name == "m").get
    assert(m.sym.isInstanceOf[ArraySymbol])
    val sym = m.sym.asInstanceOf[ArraySymbol]
    assert(sym.indexWidth == 5)
    assert(sym.dataWidth == 8)
    assert(m.init.isEmpty)
    //assert(m.next.get.toString.contains("m[m.w.addr := m.w.data]"))
    assert(m.next.get.toString == "m[m.w.addr := m.w.data]")
  }

  it should "support scalar initialization of a memory to 0" in {
    val sys = toSys(memCircuit(), memInit = Map("m" -> MemoryScalarInit(0)))
    val m = sys.states.find(_.sym.name == "m").get
    assert(m.init.isDefined)
    assert(m.init.get.toString == "([8'b0] x 32)")
  }

  it should "support scalar initialization of a memory to 127" in {
    val sys = toSys(memCircuit(31), memInit = Map("m" -> MemoryScalarInit(127)))
    val m = sys.states.find(_.sym.name == "m").get
    assert(m.init.isDefined)
    assert(m.init.get.toString == "([8'b1111111] x 32)")
  }

  it should "support array initialization of a memory to Seq(0, 1, 2, 3)" in {
    val sys = toSys(memCircuit(4), memInit = Map("m" -> MemoryArrayInit(Seq(0, 1, 2, 3))))
    val m = sys.states.find(_.sym.name == "m").get
    assert(m.init.isDefined)
    assert(m.init.get.toString == "([8'b0] x 4)[2'b1 := 8'b1][2'b10 := 8'b10][2'b11 := 8'b11]")
  }

  it should "support array initialization of a memory to Seq(1, 0, 1, 0)" in {
    val sys = toSys(memCircuit(4), memInit = Map("m" -> MemoryArrayInit(Seq(1, 0, 1, 0))))
    val m = sys.states.find(_.sym.name == "m").get
    assert(m.init.isDefined)
    assert(m.init.get.toString == "([8'b1] x 4)[2'b1 := 8'b0][2'b11 := 8'b0]")
  }

  it should "support array initialization of a memory to Seq(1, 0, 0, 0)" in {
    val sys = toSys(memCircuit(4), memInit = Map("m" -> MemoryArrayInit(Seq(1, 0, 0, 0))))
    val m = sys.states.find(_.sym.name == "m").get
    assert(m.init.isDefined)
    assert(m.init.get.toString == "([8'b0] x 4)[2'b0 := 8'b1]")
  }

  it should "support array initialization from a file" ignore {
    assert(false, "TODO")
  }

  it should "support memories with registered read port" in {
    def src(readUnderWrite: String) =
      s"""circuit m:
         |  module m:
         |    input reset : UInt<1>
         |    input clock : Clock
         |    input addr : UInt<5>
         |    input in : UInt<8>
         |    output out : UInt<8>
         |
         |    mem m:
         |      data-type => UInt<8>
         |      depth => 32
         |      reader => r
         |      writer => w1, w2
         |      read-latency => 1
         |      write-latency => 1
         |      read-under-write => $readUnderWrite
         |
         |    m.w1.clk <= clock
         |    m.w1.mask <= UInt(1)
         |    m.w1.en <= UInt(1)
         |    m.w1.data <= in
         |    m.w1.addr <= addr
         |    m.w2.clk <= clock
         |    m.w2.mask <= UInt(1)
         |    m.w2.en <= UInt(1)
         |    m.w2.data <= in
         |    m.w2.addr <= addr
         |
         |    m.r.clk <= clock
         |    m.r.en <= UInt(1)
         |    out <= m.r.data
         |    m.r.addr <= addr
         |
         |""".stripMargin

    val oldValue = toSys(src("old"))
    val oldMData = oldValue.states.find(_.sym.name == "m.r.data").get
    assert(oldMData.sym.toString == "m.r.data")
    assert(oldMData.next.get.toString == "m[m.r.addr]", "we just need to read the current value")
    val readDataSignal = oldValue.signals.find(_.name == "m.r.data")
    assert(readDataSignal.isEmpty, s"${readDataSignal.map(_.toString)} should not exist")

    val undefinedValue = toSys(src("undefined"))
    val undefinedMData = undefinedValue.states.find(_.sym.name == "m.r.data").get
    assert(undefinedMData.sym.toString == "m.r.data")
    val undefined = "RANDOM.m_r_read_under_write_undefined"
    assert(
      undefinedMData.next.get.toString ==
        s"ite(or(eq(m.r.addr, m.w1.addr), eq(m.r.addr, m.w2.addr)), $undefined, m[m.r.addr])",
      "randomize result if there is a write"
    )
  }

  it should "support memories with potential write-write conflicts" in {
    val src =
      s"""circuit m:
         |  module m:
         |    input reset : UInt<1>
         |    input clock : Clock
         |    input addr : UInt<5>
         |    input in : UInt<8>
         |    output out : UInt<8>
         |
         |    mem m:
         |      data-type => UInt<8>
         |      depth => 32
         |      reader => r
         |      writer => w1, w2
         |      read-latency => 1
         |      write-latency => 1
         |      read-under-write => undefined
         |
         |    m.w1.clk <= clock
         |    m.w1.mask <= UInt(1)
         |    m.w1.en <= UInt(1)
         |    m.w1.data <= in
         |    m.w1.addr <= addr
         |    m.w2.clk <= clock
         |    m.w2.mask <= UInt(1)
         |    m.w2.en <= UInt(1)
         |    m.w2.data <= in
         |    m.w2.addr <= addr
         |
         |    m.r.clk <= clock
         |    m.r.en <= UInt(1)
         |    out <= m.r.data
         |    m.r.addr <= addr
         |
         |""".stripMargin

    val sys = toSys(src)
    val m = sys.states.find(_.sym.name == "m").get

    val regularUpdate = "m[m.w1.addr := m.w1.data][m.w2.addr := m.w2.data]"
    val collision = "eq(m.w1.addr, m.w2.addr)"
    val collisionUpdate = "m[m.w1.addr := RANDOM.m_w1_write_write_collision]"

    assert(m.next.get.toString == s"ite($collision, $collisionUpdate, $regularUpdate)")
  }

  it should "model invalid signals as inputs" in {
    // if a signal is invalid, it could take on an arbitrary value in that cycle
    val src =
      """circuit m:
        |  module m:
        |    input en : UInt<1>
        |    output o : UInt<8>
        |    o is invalid
        |    when en:
        |      o <= UInt<8>(0)
        |""".stripMargin
    val sys = toSys(src)
    assert(sys.inputs.length == 2)
    val random = sys.inputs.filter(_.name.contains("RANDOM"))
    assert(random.length == 1)
    assert(random.head.width == 8)
  }

  it should "ignore assignments, ports, wires and nodes of clock type" in {
    // The transformation relies on the assumption that everything is connected to a single global clock
    // thus any clock ports, wires, nodes and connects should be ignored.
    val src =
      """circuit m:
        |  module m:
        |    input clk : Clock
        |    output o : Clock
        |    wire w: Clock
        |    node x = w
        |    o <= x
        |    w <= clk
        |""".stripMargin
    val sys = toSys(src)
    assert(sys.inputs.isEmpty, "Clock inputs should be ignored.")
    assert(sys.outputs.isEmpty, "Clock outputs should be ignored.")
    assert(sys.signals.isEmpty, "Connects of clock type should be ignored.")
  }

  it should "treat clock outputs of submodules like a clock input" in {
    // Since we treat any remaining submodules (that have not been inlined) as blackboxes, a clock output
    // is like a clock input to our module.
    val src =
      """circuit m:
        |  module c:
        |    output clk: Clock
        |    clk is invalid
        |  module m:
        |    input clk : Clock
        |    inst c of c
        |""".stripMargin
    val err = intercept[MultiClockException] {
      toSys(src)
    }
    assert(err.getMessage.contains("clk, c.clk"))
  }

  it should "throw an error on async reset" in {
    val err = intercept[AsyncResetException] {
      toSys(
        """circuit m:
          |  module m:
          |    input reset : AsyncReset
          |""".stripMargin
      )
    }
    assert(err.getMessage.contains("reset"))
  }

  it should "throw an error on casting to async reset" in {
    val err = intercept[AssertionError] {
      toSys(
        """circuit m:
          |  module m:
          |    input reset : UInt<1>
          |    node async = asAsyncReset(reset)
          |""".stripMargin
      )
    }
    assert(err.getMessage.contains("reset"))
  }

  it should "throw an error on multiple clocks" in {
    val err = intercept[MultiClockException] {
      toSys(
        """circuit m:
          |  module m:
          |    input clk1 : Clock
          |    input clk2 : Clock
          |""".stripMargin
      )
    }
    assert(err.getMessage.contains("clk1, clk2"))
  }

  it should "throw an error on using a clock as uInt" in {
    // While this could potentially be supported in the future, for now we do not allow
    // a clock to be used for anything besides updating registers and memories.
    val err = intercept[AssertionError] {
      toSys(
        """circuit m:
          |  module m:
          |    input clk : Clock
          |    output o : UInt<1>
          |    o <= asUInt(clk)
          |
          |""".stripMargin
      )
    }
    assert(err.getMessage.contains("clk"))
  }
}
