package chisel3.std.tilelink

import chisel3._
import chisel3.experimental.{SerializableModule, SerializableModuleParameter}
import org.scalatest.flatspec.AnyFlatSpec

/**
  * Parameter for TLRam
  */
case class TLRamParameter(
  busParam: TileLinkLinkParameter)
    extends SerializableModuleParameter {
  require(!busParam.hasBCEChannels, "TLRam does not support TL-C")
}

/**
  * A simple RAM attached to TileLink bus
  */
class TLRam(val parameter: TLRamParameter) extends Module with SerializableModule[TLRamParameter] {
  val busIO: TLBundle = IO(Flipped(new TLBundle(parameter.busParam)))

  val addrBits: Int = parameter.busParam.addressWidth
  val dataBits: Int = parameter.busParam.dataWidth

  val mem = SyncReadMem(BigInt(1) << addrBits, Vec(dataBits / 8, UInt(8.W)))

  val a_read = Wire(Bool())
  val a_wen = Wire(Bool())
  val a_wdata = Wire(Vec(dataBits / 8, UInt(8.W)))
  val a_mask = Wire(Vec(dataBits / 8, Bool()))
  val a_addr = Wire(UInt(addrBits.W))
  val a_ready = Wire(Bool())

  val d_mem_out = Reg(UInt(dataBits.W))
  val d_resp = Reg(UInt(OpCode.width))
  val d_source = Reg(UInt(parameter.busParam.sourceWidth.W))
  val d_size = Reg(UInt(parameter.busParam.sizeWidth.W))
  val d_valid = RegInit(false.B)

  // Control path
  a_ready := busIO.d.ready || !d_valid
  d_valid := busIO.a.fire || (d_valid & !busIO.d.ready)

  // Data path
  a_read := busIO.a.bits.opcode === OpCode.Get
  a_wen := busIO.a.bits.opcode === OpCode.PutFullData || busIO.a.bits.opcode === OpCode.PutPartialData
  a_wdata := busIO.a.bits.data.asTypeOf(a_wdata)
  a_mask := busIO.a.bits.mask.asBools
  a_addr := busIO.a.bits.address

  when(busIO.a.fire) {
    d_mem_out := mem.read(a_addr, a_read).asUInt
    when(a_wen) {
      mem.write(a_addr, a_wdata, a_mask)
      d_resp := OpCode.AccessAck
    }.otherwise {
      d_resp := OpCode.AccessAckData
    }
    // Pass through other metadata
    d_source := busIO.a.bits.source
    d_size := busIO.a.bits.size
  }

  // Bus connection
  busIO.a.ready := a_ready
  busIO.d.valid := d_valid
  busIO.d.bits.opcode := d_resp
  busIO.d.bits.param := Param.tieZero
  busIO.d.bits.size := d_size
  busIO.d.bits.source := d_source
  busIO.d.bits.sink := DontCare
  busIO.d.bits.denied := false.B
  busIO.d.bits.corrupt := false.B
  busIO.d.bits.data := d_mem_out
}

class TLRamTester extends AnyFlatSpec {
  val addressWidth = 8
  val sourceWidth = 4
  val sinkWidth = 4
  val dataWidth = 32
  val sizeWidth = 2
  val linkParameter =
    TileLinkLinkParameter(addressWidth, sourceWidth, sinkWidth, dataWidth, sizeWidth, hasBCEChannels = false)

  "TLRam" should "elaborate TileLink bus connection" in {
    chisel3.stage.ChiselStage.emitChirrtl(new TLRam(TLRamParameter(linkParameter)))
  }
}
