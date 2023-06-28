// SPDX-License-Identifier: Apache-2.0

package chiselTests

import circt.stage.ChiselStage
import chisel3._
import chisel3.testers.BasicTester
import chisel3.util._

class MemVecTester extends BasicTester {
  val mem = Mem(2, Vec(2, UInt(8.W)))

  // Circuit style tester is definitely the wrong abstraction here
  val (cnt, wrap) = Counter(true.B, 2)
  mem(0)(0) := 1.U

  when(cnt === 1.U) {
    assert(mem.read(0.U)(0) === 1.U)
    stop()
  }
}

class SyncReadMemTester extends BasicTester {
  val (cnt, _) = Counter(true.B, 5)
  val mem = SyncReadMem(2, UInt(2.W))
  val rdata = mem.read(cnt - 1.U, cnt =/= 0.U)

  switch(cnt) {
    is(0.U) { mem.write(cnt, 3.U) }
    is(1.U) { mem.write(cnt, 2.U) }
    is(2.U) { assert(rdata === 3.U) }
    is(3.U) { assert(rdata === 2.U) }
    is(4.U) { stop() }
  }
}

class SyncReadMemWriteCollisionTester extends BasicTester {
  val (cnt, _) = Counter(true.B, 5)

  // Write-first
  val m0 = SyncReadMem(2, UInt(2.W), SyncReadMem.WriteFirst)
  val rd0 = m0.read(cnt)
  m0.write(cnt, cnt)

  // Read-first
  val m1 = SyncReadMem(2, UInt(2.W), SyncReadMem.ReadFirst)
  val rd1 = m1.read(cnt)
  m1.write(cnt, cnt)

  // Read data from address 0
  when(cnt === 3.U) {
    assert(rd0 === 2.U)
    assert(rd1 === 0.U)
  }

  when(cnt === 4.U) {
    stop()
  }
}

class SyncReadMemWithZeroWidthTester extends BasicTester {
  val (cnt, _) = Counter(true.B, 3)
  val mem = SyncReadMem(2, UInt(0.W))
  val rdata = mem.read(0.U, true.B)

  switch(cnt) {
    is(1.U) { assert(rdata === 0.U) }
    is(2.U) { stop() }
  }
}

// TODO this can't actually simulate with FIRRTL behavioral mems
class HugeSMemTester(size: BigInt) extends BasicTester {
  val (cnt, _) = Counter(true.B, 5)
  val mem = SyncReadMem(size, UInt(8.W))
  val rdata = mem.read(cnt - 1.U, cnt =/= 0.U)

  switch(cnt) {
    is(0.U) { mem.write(cnt, 3.U) }
    is(1.U) { mem.write(cnt, 2.U) }
    is(2.U) { assert(rdata === 3.U) }
    is(3.U) { assert(rdata === 2.U) }
    is(4.U) { stop() }
  }
}
class HugeCMemTester(size: BigInt) extends BasicTester {
  val (cnt, _) = Counter(true.B, 5)
  val mem = Mem(size, UInt(8.W))
  val rdata = mem.read(cnt)

  switch(cnt) {
    is(0.U) { mem.write(cnt, 3.U) }
    is(1.U) { mem.write(cnt, 2.U) }
    is(2.U) { assert(rdata === 3.U) }
    is(3.U) { assert(rdata === 2.U) }
    is(4.U) { stop() }
  }
}

class SyncReadMemBundleTester extends BasicTester {
  val (cnt, _) = Counter(true.B, 5)
  val tpe = new Bundle {
    val foo = UInt(2.W)
  }
  val mem = SyncReadMem(2, tpe)
  val rdata = mem.read(cnt - 1.U, cnt =/= 0.U)

  switch(cnt) {
    is(0.U) {
      val w = Wire(tpe)
      w.foo := 3.U
      mem.write(cnt, w)
    }
    is(1.U) {
      val w = Wire(tpe)
      w.foo := 2.U
      mem.write(cnt, w)
    }
    is(2.U) { assert(rdata.foo === 3.U) }
    is(3.U) { assert(rdata.foo === 2.U) }
    is(4.U) { stop() }
  }
}

class MemBundleTester extends BasicTester {
  val tpe = new Bundle {
    val foo = UInt(2.W)
  }
  val mem = Mem(2, tpe)

  // Circuit style tester is definitely the wrong abstraction here
  val (cnt, wrap) = Counter(true.B, 2)
  mem(0) := {
    val w = Wire(tpe)
    w.foo := 1.U
    w
  }

  when(cnt === 1.U) {
    assert(mem.read(0.U).foo === 1.U)
    stop()
  }
}

private class TrueDualPortMemoryIO(val addrW: Int, val dataW: Int) extends Bundle {
  require(addrW > 0, "address width must be greater than 0")
  require(dataW > 0, "data width must be greater than 0")

  val clka = Input(Clock())
  val ena = Input(Bool())
  val wea = Input(Bool())
  val addra = Input(UInt(addrW.W))
  val dia = Input(UInt(dataW.W))
  val doa = Output(UInt(dataW.W))

  val clkb = Input(Clock())
  val enb = Input(Bool())
  val web = Input(Bool())
  val addrb = Input(UInt(addrW.W))
  val dib = Input(UInt(dataW.W))
  val dob = Output(UInt(dataW.W))
}

private class TrueDualPortMemory(addrW: Int, dataW: Int) extends RawModule {
  val io = IO(new TrueDualPortMemoryIO(addrW, dataW))
  val ram = SyncReadMem(1 << addrW, UInt(dataW.W))

  // Port a
  withClock(io.clka) {
    io.doa := DontCare
    when(io.ena) {
      when(io.wea) {
        ram(io.addra) := io.dia
      }
      io.doa := ram(io.addra)
    }
  }

  // Port b
  withClock(io.clkb) {
    io.dob := DontCare
    when(io.enb) {
      when(io.web) {
        ram(io.addrb) := io.dib
      }
      io.dob := ram(io.addrb)
    }
  }
}

class MemReadWriteTester extends BasicTester {
  val (cnt, _) = Counter(true.B, 6)
  val mem = SyncReadMem(2, UInt(2.W))

  // The address to write to, alternating between 0 and 1 each cycle
  val address = Wire(UInt())
  address := DontCare

  // The data to write into the read-write port
  val wdata = Wire(UInt(8.W))
  wdata := DontCare

  // Enable signal
  val enable = Wire(Bool())
  enable := true.B // By default, memory access is on

  // Write signal
  val isWrite = Wire(Bool())
  isWrite := false.B // By default, writes are off

  val rdata = mem.readWrite(address, wdata, enable, isWrite)

  switch(cnt) {
    is(0.U) { // Cycle 1: Write 3.U to address 0
      address := 0.U
      enable := true.B
      isWrite := true.B
      wdata := 3.U
    }
    is(1.U) { // Cycle 2: Write 2.U to address 1
      address := 1.U
      enable := true.B
      isWrite := true.B
      wdata := 2.U
    }
    is(2.U) { // Cycle 3: Read from address 0 (data returned next cycle)
      address := 0.U
      enable := true.B
      isWrite := false.B
    }
    is(3.U) { // Cycle 4: Expect RDWR port to contain 3.U, then read from address 1
      address := 1.U
      enable := true.B
      isWrite := false.B
      assert(rdata === 3.U)
    }
    is(4.U) { // Cycle 5: Expect rdata to contain 2.U
      assert(rdata === 2.U)
    }
    is(5.U) { // Cycle 6: Stop
      stop()
    }
  }
}

class MemMaskedReadWriteTester extends BasicTester {
  val (cnt, _) = Counter(true.B, 11)
  val mem = SyncReadMem(2, Vec(4, UInt(8.W)))

  // The address to write to, alternating between 0 and 1 each cycle
  val address = Wire(UInt())
  address := DontCare

  // The data to write into the read-write port
  val wdata = Wire(Vec(4, UInt(8.W)))
  wdata := DontCare

  // The bytemask used for masking readWrite
  val mask = Wire(Vec(4, Bool()))
  mask := DontCare

  // Enable signal
  val enable = Wire(Bool())
  enable := true.B // By default, memory access is on

  // Write signal
  val isWrite = Wire(Bool())
  isWrite := false.B // By default, writes are off

  val rdata = mem.readWrite(address, wdata, mask, enable, isWrite)

  switch(cnt) {
    is(0.U) { // Cycle 1: Write (1.U, 2.U, 3.U, 4.U) with mask (1, 1, 1, 1) to address 0
      address := 0.U
      enable := true.B
      isWrite := true.B
      mask := VecInit.fill(4)(true.B)
      wdata := VecInit(1.U, 2.U, 3.U, 4.U)
    }
    is(1.U) { // Cycle 2: Write (5.U, 6.U, 7.U, 8.U) with mask (1, 1, 1, 1) to address 1
      address := 1.U
      enable := true.B
      isWrite := true.B
      mask := VecInit.fill(4)(true.B)
      wdata := VecInit(5.U, 6.U, 7.U, 8.U)
    }
    is(2.U) { // Cycle 3: Read from address 0 (data returned next cycle)
      address := 0.U
      enable := true.B
      isWrite := false.B
    }
    is(3.U) { // Cycle 4: Expect RDWR port to contain (1.U, 2.U, 3.U, 4.U), then read from address 1
      assert(rdata === VecInit(1.U, 2.U, 3.U, 4.U))

      address := 1.U
      enable := true.B
      isWrite := false.B
    }
    is(4.U) { // Cycle 5: Expect rdata to contain (5.U, 6.U, 7.U, 8.U)
      assert(rdata === VecInit(5.U, 6.U, 7.U, 8.U))
    }
    is(5.U) { // Cycle 6: Write (0.U, - , - , 0.U) with mask (1, 0, 0, 1) to address 0
      address := 0.U
      enable := true.B
      isWrite := true.B
      mask := VecInit(true.B, false.B, false.B, true.B)
      // Bogus values for 2nd and 3rd indices to make sure they aren't actually written
      wdata := VecInit(0.U, 100.U, 100.U, 0.U)
    }
    is(6.U) { // Cycle 7: Write (- , 0.U , 0.U , -) with mask (0, 1, 1, 0) to address 1
      address := 1.U
      enable := true.B
      isWrite := true.B
      mask := VecInit(false.B, true.B, true.B, false.B)
      // Bogus values for 1st and 4th indices to make sure they aren't actually written
      wdata := VecInit(100.U, 0.U, 0.U, 100.U)
    }
    is(7.U) { // Cycle 8: Read from address 0 (data returned next cycle)
      address := 0.U
      enable := true.B
      isWrite := false.B
    }
    is(8.U) { // Cycle 9: Expect RDWR port to contain (0.U, 2.U, 3.U, 0.U), then read from address 1
      // NOT (0.U, 100.U, 100.U, 0.U)
      assert(rdata === VecInit(0.U, 2.U, 3.U, 0.U))

      address := 1.U
      enable := true.B
      isWrite := false.B
    }
    is(9.U) { // Cycle 10: Expect rdata to contain (5.U, 0.U, 0.U, 8.U)
      // NOT (100.U, 0.U, 0.U, 100.U)
      assert(rdata === VecInit(5.U, 0.U, 0.U, 8.U))
    }
    is(10.U) { // Cycle 11: Stop
      stop()
    }
  }
}

class MemorySpec extends ChiselPropSpec {
  property("Mem of Vec should work") {
    assertTesterPasses { new MemVecTester }
  }

  property("SyncReadMem should work") {
    assertTesterPasses { new SyncReadMemTester }
  }

  property("SyncReadMems of Bundles should work") {
    assertTesterPasses { new SyncReadMemBundleTester }
  }

  property("Mems of Bundles should work") {
    assertTesterPasses { new MemBundleTester }
  }

  //TODO: SFC->MFC, this test is ignored because the read-under-write specifiers are not emitted to work with MFC
  ignore("SyncReadMem write collision behaviors should work") {
    assertTesterPasses { new SyncReadMemWriteCollisionTester }
  }

  property("SyncReadMem should work with zero width entry") {
    assertTesterPasses { new SyncReadMemWithZeroWidthTester }
  }

  property("SyncReadMems should be able to have an explicit number of read-write ports") {
    // Check if there is exactly one MemReadWrite port (TODO: extend to Nr/Nw?)
    val chirrtl = ChiselStage.emitCHIRRTL(new MemReadWriteTester)
    chirrtl should include(s"rdwr mport rdata = mem[_rdata_T_1], clock")

    // Check read/write logic
    assertTesterPasses { new MemReadWriteTester }
  }

  property("SyncReadMem masked read-writes should work") {
    // Check if there is exactly one MemReadWrite port (TODO: extend to Nr/Nw?)
    val chirrtl = ChiselStage.emitCHIRRTL(new MemMaskedReadWriteTester)
    chirrtl should include(s"rdwr mport rdata = mem[_rdata_T_1], clock")

    // Check read/write logic
    assertTesterPasses { new MemMaskedReadWriteTester }
  }

  property("Massive memories should be emitted in Verilog") {
    val addrWidth = 65
    val size = BigInt(1) << addrWidth
    val smem = ChiselStage.emitCHIRRTL(new HugeSMemTester(size))
    smem should include(s"smem mem : UInt<8> [$size]")
    val cmem = ChiselStage.emitCHIRRTL(new HugeCMemTester(size))
    cmem should include(s"cmem mem : UInt<8> [$size]")
  }

  property("Implicit conversions with Mem indices should work") {
    """
      |import chisel3._
      |import chisel3.util.ImplicitConversions._
      |class MyModule extends Module {
      |  val io = IO(new Bundle {})
      |  val mem = Mem(32, UInt(8.W))
      |  mem(0) := 0.U
      |}
      |""".stripMargin should compile
  }

  property("memories in modules without implicit clock should compile without warning or error") {
    ChiselStage.emitCHIRRTL(new TrueDualPortMemory(4, 32))
  }

  property("Memories can have addresses driven before their declarations") {
    class TestModule extends Module {
      val io = IO(new Bundle {
        val rdEnable = Input(Bool())
        val writeData = Input(UInt(2.W))
        val mrwWriteData = Input(Vec(2, UInt(2.W)))
        val mrwWriteMask = Input(Vec(2, Bool()))

        val rwEnable = Input(Bool())
        val rwIsWrite = Input(Bool())
        val mrwIsWrite = Input(Bool())

        val rdReadValue = Output(UInt(2.W))
        val rwReadValue = Output(UInt(2.W))
        val mrwReadValue = Output(Vec(2, UInt(2.W)))
      })

      // Address value declared and driven before the SyncReadMem declaration.
      // This is OK in Chisel, with the caveat that an intermediate wire is
      // generated with the address after a memory port is instantiated -- if
      // not then SFC and firtool are unable to infer the address value of the
      // memory port correctly and results in an error
      val addr = Wire(UInt(2.W))
      addr := 0.U

      val mem = SyncReadMem(4, UInt(2.W))
      val vecMem = SyncReadMem(4, Vec(2, UInt(2.W)))

      // Should elaborate correctly
      io.rdReadValue := mem.read(addr, io.rdEnable)
      // Should elaborate correctly
      mem.write(addr, io.writeData)
      // Should elaborate correctly
      io.rwReadValue := mem.readWrite(addr, io.writeData, io.rwEnable, io.rwIsWrite)
      // Should elaborate correctly
      io.mrwReadValue := vecMem.readWrite(addr, io.mrwWriteData, io.mrwWriteMask, io.rwEnable, io.rwIsWrite)
    }
    ChiselStage.emitSystemVerilog(new TestModule)
  }
}

class SRAMSpec extends ChiselFunSpec {
  describe("SRAM") {
    val portCombos: Seq[(Int, Int, Int)] =
      for {
        numRD <- 0 until 3
        numWR <- 0 until 3
        numRW <- 0 until 3
        if (numRD + numWR + numRW) > 0
        if (numRD + numRW) > 0
        if (numWR + numRW) > 0
      } yield (numRD, numWR, numRW)

    portCombos.foreach {
      case (numRD, numWR, numRW) =>
        val portedness: String = {
          val rdPorts: String = if (numRD > 0) s"${numRD}R" else ""
          val wrPorts: String = if (numWR > 0) s"${numWR}W" else ""
          val rwPorts: String = if (numRW > 0) s"${numRW}RW" else ""

          s"$rdPorts$wrPorts$rwPorts"
        }
        it(s"should generate a $portedness memory") {
          class TestModule(val rd: Int, val wr: Int, val rw: Int) extends Module {
            val mem = SRAM(32, UInt(8.W), rd, wr, rw)

            dontTouch(mem)

            for (i <- 0 until rd) {
              mem.readPorts(i) := DontCare
            }
            for (i <- 0 until wr) {
              mem.writePorts(i) := DontCare
            }
            for (i <- 0 until rw) {
              mem.readwritePorts(i) := DontCare
            }
          }
          val chirrtl = ChiselStage.emitCHIRRTL(new TestModule(numRD, numWR, numRW), args = Array("--full-stacktrace"))

          // Check that the chirrtl ports actually exist and the signals
          // are properly connected
          for (rd <- 0 until numRD) {
            val rdPortName = s"mem_out_readPorts_${rd}_data_MPORT"
            chirrtl should include(s"when mem.readPorts[$rd].enable")
            chirrtl should include(s"read mport $rdPortName")
            chirrtl should include(s"connect mem.readPorts[$rd].data, $rdPortName")
          }

          for (wr <- 0 until numWR) {
            val wrPortName = s"mem_MPORT${if (wr == 0) "" else s"_$wr"}"
            chirrtl should include(s"when mem.writePorts[$wr].enable")
            chirrtl should include(s"write mport $wrPortName")
            chirrtl should include(s"connect $wrPortName, mem.writePorts[$wr].data")
          }

          for (rw <- 0 until numRW) {
            val rwPortName = s"mem_out_readwritePorts_${rw}_readData_MPORT"
            chirrtl should include(s"when mem.readwritePorts[$rw].enable")
            chirrtl should include(s"rdwr mport $rwPortName")
            chirrtl should include(s"when mem.readwritePorts[$rw].isWrite")
            chirrtl should include(s"connect $rwPortName, mem.readwritePorts[$rw].writeData")
          }
        }
    }

    it(s"should support masking with Vec-valued data") {
      class TestModule(val wr: Int, val rw: Int) extends Module {
        val mem = SRAM.masked(32, Vec(3, UInt(8.W)), 0, wr, rw)

        dontTouch(mem)

        for (i <- 0 until wr) {
          mem.writePorts(i) := DontCare
        }
        for (i <- 0 until rw) {
          mem.readwritePorts(i) := DontCare
        }
      }
      val chirrtl = ChiselStage.emitCHIRRTL(new TestModule(1, 1), args = Array("--full-stacktrace"))

      chirrtl should include(
        "writePorts : { flip address : UInt<6>, flip enable : UInt<1>, flip data : UInt<8>[3], flip mask : UInt<1>[3]}[1]"
      )
      chirrtl should include(
        "readwritePorts : { flip address : UInt<6>, flip enable : UInt<1>, flip isWrite : UInt<1>, readData : UInt<8>[3], flip writeData : UInt<8>[3], flip mask : UInt<1>[3]}[1]"
      )

      for (i <- 0 until 3) {
        chirrtl should include(s"when mem.writePorts[0].mask[$i]")
        chirrtl should include(s"connect mem_MPORT[$i], mem.writePorts[0].data[$i]")

        chirrtl should include(s"when mem.readwritePorts[0].mask[$i]")
        chirrtl should include(
          s"connect mem_out_readwritePorts_0_readData_MPORT[$i], mem.readwritePorts[0].writeData[$i]"
        )
      }
    }

    it(s"should support multiple clocks driving different ports") {
      class TestModule extends Module {
        val (counter, _) = Counter(true.B, 11)

        val readClocks = IO(Input(Vec(3, Clock())))
        val writeClocks = IO(Input(Vec(3, Clock())))
        val readwriteClocks = IO(Input(Vec(3, Clock())))

        val mem = SRAM(
          32,
          Vec(3, UInt(8.W)),
          readClocks,
          writeClocks,
          readwriteClocks
        )

        dontTouch(mem)

        for (i <- 0 until 3) {
          mem.readPorts(i) := DontCare
          mem.writePorts(i) := DontCare
          mem.readwritePorts(i) := DontCare
        }
      }
      val chirrtl = ChiselStage.emitCHIRRTL(new TestModule, args = Array("--full-stacktrace"))

      for (i <- 0 until 3) {
        val wrIndexSuffix = if (i == 0) "" else s"_$i"

        chirrtl should include(
          s"read mport mem_out_readPorts_${i}_data_MPORT = mem_mem[_mem_out_readPorts_${i}_data_T], readClocks[${i}]"
        )
        chirrtl should include(
          s"write mport mem_MPORT${wrIndexSuffix} = mem_mem[_mem_T${wrIndexSuffix}], writeClocks[${i}]"
        )
        chirrtl should include(
          s"rdwr mport mem_out_readwritePorts_${i}_readData_MPORT = mem_mem[_mem_out_readwritePorts_${i}_readData_T], readwriteClocks[${i}]"
        )
      }
    }
  }

  describe("Read-only SRAM") {
    it(s"should error") {
      class TestModule extends Module {
        val mem = SRAM(32, Vec(3, UInt(8.W)), 1, 0, 0)
      }
      intercept[Exception] {
        ChiselStage.emitCHIRRTL(new TestModule, args = Array("--full-stacktrace"))
      }
    }
  }

  describe("Write-only SRAM") {
    it(s"should error") {
      class TestModule extends Module {
        val mem = SRAM(32, Vec(3, UInt(8.W)), 0, 1, 0)
      }
      intercept[Exception] {
        ChiselStage.emitCHIRRTL(new TestModule, args = Array("--full-stacktrace"))
      }
    }
  }
}
