// See LICENSE for license details.

package chiselTests

class CompatibiltySpec extends ChiselFlatSpec  {
  import Chisel._

  behavior of "Chisel compatibility layer"

  it should "contain the following definitions" in {
    INPUT == chisel3.core.Direction.Input should be(true)
    OUTPUT == chisel3.core.Direction.Output should be(true)
    NODIR == chisel3.core.Direction.Unspecified should be(true)
    Wire == chisel3.core.Wire should be(true)
    Clock == chisel3.core.Clock should be(true)
    Vec == chisel3.core.Vec should be(true)
    Chisel.assert == chisel3.core.assert should be(true)
    stop == chisel3.core.stop should be(true)

    Bits == chisel3.core.Bits should be(true)
    UInt == chisel3.core.UInt should be(true)
    SInt == chisel3.core.SInt should be(true)
    Bool == chisel3.core.Bool should be(true)
    Mux == chisel3.core.Mux should be(true)

    Mem == chisel3.core.Mem should be(true)
    SeqMem == chisel3.core.SeqMem should be(true)

    Module == chisel3.core.Module should be(true)

    printf == chisel3.core.printf should be(true)

    Reg == chisel3.core.Reg should be(true)

    when == chisel3.core.when should be(true)

    Driver == chisel3.Driver should be(true)
    ImplicitConversions == chisel3.util.ImplicitConversions should be(true)
    chiselMain == chisel3.compatibility.chiselMain should be(true)
    throwException == chisel3.compatibility.throwException should be(true)
    debug == chisel3.compatibility.debug should be(true)

    testers.TesterDriver == chisel3.testers.TesterDriver should be(true)
    log2Up == chisel3.util.log2Up should be(true)
    log2Ceil == chisel3.util.log2Ceil should be(true)
    log2Down == chisel3.util.log2Down should be(true)
    log2Floor == chisel3.util.log2Floor should be(true)
    isPow2 == chisel3.util.isPow2 should be(true)

    BitPat == chisel3.util.BitPat should be(true)

    FillInterleaved == chisel3.util.FillInterleaved should be(true)
    PopCount == chisel3.util.PopCount should be(true)
    Fill == chisel3.util.Fill should be(true)
    Reverse == chisel3.util.Reverse should be(true)

    Cat == chisel3.util.Cat should be(true)

    Log2 == chisel3.util.Log2 should be(true)

    unless == chisel3.util.unless should be(true)
    is == chisel3.util.is should be(true)
    switch == chisel3.util.switch should be(true)

    Counter == chisel3.util.Counter should be(true)

    DecoupledIO == chisel3.util.Decoupled should be(true)
    Decoupled == chisel3.util.Decoupled should be(true)
    Queue == chisel3.util.Queue should be(true)

    Enum == chisel3.util.Enum should be(true)

    LFSR16 == chisel3.util.LFSR16 should be(true)

    ListLookup == chisel3.util.ListLookup should be(true)
    Lookup == chisel3.util.Lookup should be(true)

    Mux1H == chisel3.util.Mux1H should be(true)
    PriorityMux == chisel3.util.PriorityMux should be(true)
    MuxLookup == chisel3.util.MuxLookup should be(true)
    MuxCase == chisel3.util.MuxCase should be(true)

    OHToUInt == chisel3.util.OHToUInt should be(true)
    PriorityEncoder == chisel3.util.PriorityEncoder should be(true)
    UIntToOH == chisel3.util.UIntToOH should be(true)
    PriorityEncoderOH == chisel3.util.PriorityEncoderOH should be(true)

    RegNext == chisel3.util.RegNext should be(true)
    RegInit == chisel3.util.RegInit should be(true)
    RegEnable == chisel3.util.RegEnable should be(true)
    ShiftRegister == chisel3.util.ShiftRegister should be(true)

    Valid == chisel3.util.Valid should be(true)
    Pipe == chisel3.util.Pipe should be(true)

  }

  // Verify we can elaborate a design expressed in Chisel2
  class Chisel2CompatibleRisc extends Module {
    val io = new Bundle {
      val isWr   = Bool(INPUT)
      val wrAddr = UInt(INPUT, 8)
      val wrData = Bits(INPUT, 32)
      val boot   = Bool(INPUT)
      val valid  = Bool(OUTPUT)
      val out    = Bits(OUTPUT, 32)
    }
    val file = Mem(Bits(width = 32), 256)
    val code = Mem(Bits(width = 32), 256)
    val pc   = Reg(init=UInt(0, 8))

    val add_op :: imm_op :: Nil = Enum(Bits(width = 8), 2)

    val inst = code(pc)
    val op   = inst(31,24)
    val rci  = inst(23,16)
    val rai  = inst(15, 8)
    val rbi  = inst( 7, 0)

    val ra = Mux(rai === Bits(0), Bits(0), file(rai))
    val rb = Mux(rbi === Bits(0), Bits(0), file(rbi))
    val rc = Wire(Bits(width = 32))

    io.valid := Bool(false)
    io.out   := Bits(0)
    rc       := Bits(0)

    when (io.isWr) {
      code(io.wrAddr) := io.wrData
    } .elsewhen (io.boot) {
      pc := UInt(0)
    } .otherwise {
      switch(op) {
        is(add_op) { rc := ra +% rb }
        is(imm_op) { rc := (rai << 8) | rbi }
      }
      io.out := rc
      when (rci === UInt(255)) {
        io.valid := Bool(true)
      } .otherwise {
        file(rci) := rc
      }
      pc := pc +% UInt(1)
    }
  }

  it should "Chisel2CompatibleRisc should elaborate" in {
    elaborate { new Chisel2CompatibleRisc }
  }

}
