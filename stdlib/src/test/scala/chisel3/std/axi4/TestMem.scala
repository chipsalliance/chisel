// SPDX-License-Identifier: Apache-2.0

package chisel3.std.axi4

import chisel3._
import chisel3.stage.ChiselStage
import chisel3.util._
import org.scalatest.flatspec.AnyFlatSpec

import scala.annotation.tailrec

/** Simple test to make sure that we can properly instantiate and elaborate the Axi4 bundles. */
class Axi4MemTests extends AnyFlatSpec {
  behavior.of("Axi4TestMem")

  it should "elaborate" in {
    val params = Axi4IOParameters(addressBits = 32, dataBits = 32, idBits = 0)
    val targetDir = "test_run_dir/Axi4TestMem_should_elaborate"
    val args = Array("-td", targetDir, "--full-stacktrace")
    (new ChiselStage).emitSystemVerilog(new Axi4TestMem(params, baseAddress = 0x1000, mask = 0xff), args)
  }
}

/** A simple Axi4 memory based on the Axi4 SRAM from rocket-chip. */
class Axi4TestMem(params: Axi4IOParameters, baseAddress: BigInt, mask: BigInt) extends Module {
  val io = IO(Flipped(new Axi4IO(params)))

  // TODO: allow 0-bit signals to be unconnected
  io.readData.bits.user := DontCare
  io.writeResponse.bits.user := DontCare

  // parameter calculations
  val beatBytes = params.dataBits / 8
  @tailrec
  private def toBitList(x: BigInt, tail: List[Boolean] = Nil): List[Boolean] =
    if (x == 0) tail.reverse else toBitList(x >> 1, ((x & 1) == 1) :: tail)
  val maskBits = toBitList(mask >> log2Ceil(beatBytes))
  val memoryEntries = BigInt(1) << maskBits.count(b => b)

  val mem = SyncReadMem(memoryEntries, Vec(beatBytes, UInt(8.W)))

  private def decodeAddress(address: UInt): (UInt, Bool) = {
    val a = Cat(maskBits.zip((address >> log2Ceil(beatBytes)).asBools).filter(_._1).map(_._2).reverse)
    val sel = ((address ^ baseAddress.U).zext & (~mask).S) === 0.S
    (a, sel)
  }
  val (r_addr, r_sel0) = decodeAddress(io.readAddress.bits.address)
  val (w_addr, w_sel0) = decodeAddress(io.writeAddress.bits.address)

  val w_full = RegInit(false.B)
  val w_id = Reg(UInt(params.idBits.W))
  val r_sel1 = RegNext(r_sel0)
  val w_sel1 = RegNext(w_sel0)

  when(io.writeResponse.fire) { w_full := false.B }
  when(io.writeAddress.fire) { w_full := true.B }

  when(io.writeAddress.fire) {
    w_id := io.writeAddress.bits.id
    w_sel1 := w_sel0
  }

  val wdata = VecInit.tabulate(beatBytes) { i => io.writeData.bits.data(8 * (i + 1) - 1, 8 * i) }
  when(io.writeAddress.fire && w_sel0) {
    mem.write(w_addr, wdata, io.writeData.bits.strobes.asBools)
  }

  io.writeResponse.valid := w_full
  io.writeAddress.ready := io.writeData.valid && (io.writeResponse.ready || !w_full)
  io.writeData.ready := io.writeAddress.valid && (io.writeResponse.ready || !w_full)

  io.writeResponse.bits.id := w_id
  when(w_sel1) {
    io.writeResponse.bits.resp := Axi4Response.Okay
  }.otherwise {
    io.writeResponse.bits.resp := Axi4Response.DecodeError
  }

  val r_full = RegInit(false.B)
  val r_id = Reg(UInt(params.idBits.W))

  when(io.readData.fire) { r_full := false.B }
  when(io.readAddress.fire) { r_full := true.B }

  when(io.readAddress.fire) {
    r_id := io.readAddress.bits.id
    r_sel1 := r_sel0
  }

  val rdata = readAndHold(mem, r_addr, io.readAddress.fire)

  io.readData.valid := r_full
  io.readAddress.ready := io.readData.ready || !r_full

  io.readData.bits.id := r_id
  when(r_sel1) {
    io.readData.bits.resp := Axi4Response.Okay
  }.otherwise {
    io.readData.bits.resp := Axi4Response.DecodeError
  }
  io.readData.bits.data := Cat(rdata.reverse)
  io.readData.bits.last := true.B
}

private object readAndHold {
  def apply[D <: Data](mem: SyncReadMem[D], addr: UInt, ren: Bool): D = {
    val data = mem.read(addr, ren)
    val enable = RegNext(ren)
    Mux(enable, data, RegEnable(data, enable))
  }
}
