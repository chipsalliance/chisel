package chisel3.util

import chisel3._

import chisel3.internal.Builder
import chisel3.experimental.SourceInfo
import chisel3.internal.sourceinfo.{MemTransform, SourceInfoTransform}
import scala.language.reflectiveCalls
import scala.language.experimental.macros

/** A bundle of signals representing a memory read port.
  *
  * @tparam tpe The data type of the memory port
  * @param addrWidth The width of the address signal
  */
class MemoryReadPort[T <: Data](tpe: T, addrWidth: Int) extends Bundle {
  val addr = Input(UInt(addrWidth.W))
  val enable = Input(Bool())
  val readValue = Output(tpe)
}

/** A bundle of signals representing a memory write port.
  *
  * @tparam tpe The data type of the memory port
  * @param addrWidth The width of the address signal
  * @param masked Whether this read/write port should have an optional mask.
  *
  * @note `masked` is only valid if tpe is a Vec; if this is not the case no mask will be initialized
  *       regardless of the value of `masked`.
  */
class MemoryWritePort[T <: Data] private[chisel3] (tpe: T, addrWidth: Int, masked: Boolean) extends Bundle {
  val addr = Input(UInt(addrWidth.W))
  val enable = Input(Bool())
  val writeValue = Input(tpe)
  val mask: Option[Vec[Bool]] = if (masked) {
    val maskSize = tpe match {
      case vec: Vec[_] => vec.size
      case _ => 0
    }
    Some(Input(Vec(maskSize, Bool())))
  } else {
    None
  }
}

/** A bundle of signals representing a memory read/write port.
  *
  * @tparam tpe The data type of the memory port
  * @param addrWidth The width of the address signal
  * @param masked Whether this read/write port should have an optional mask.
  *
  * @note `masked` is only valid if tpe is a Vec; if this is not the case no mask will be initialized
  *       regardless of the value of `masked`.
  */
class MemoryReadWritePort[T <: Data] private[chisel3] (tpe: T, addrWidth: Int, masked: Boolean) extends Bundle {
  val addr = Input(UInt(addrWidth.W))
  val enable = Input(Bool())
  val isWrite = Input(Bool())
  val readValue = Output(tpe)
  val writeValue = Input(tpe)
  val mask: Option[Vec[Bool]] = if (masked) {
    val maskSize = tpe match {
      case vec: Vec[_] => vec.size
      case _ => 0
    }
    Some(Input(Vec(maskSize, Bool())))
  } else {
    None
  }
}

/** A IO bundle of signals connecting to the ports of a wrapped `SyncReadMem`, as requested by
  * `MemInterface.apply`.
  *
  * @tparam tpe The data type of the memory port
  * @param width The width of the address wires of each port
  * @param numRd The number of read ports
  * @param numWr The number of write ports
  * @param numRdWr The number of read/write ports
  */
class MemInterface[T <: Data](tpe: T, addrWidth: Int, numRd: Int, numWr: Int, numRdWr: Int, masked: Boolean)
    extends Bundle {
  if (masked) {
    require(
      tpe.isInstanceOf[Vec[_]],
      s"masked writes require that MemInterface is instantiated with a data type of Vec (got $tpe instead)"
    )
  }
  override def typeName: String = s"MemInterface_${MemInterface.portedness(numRd, numWr, numRdWr)}"

  val rd: Vec[MemoryReadPort[T]] = Vec(numRd, new MemoryReadPort(tpe, addrWidth))
  val wr: Vec[MemoryWritePort[T]] = Vec(numWr, new MemoryWritePort(tpe, addrWidth, masked))
  val rw: Vec[MemoryReadWritePort[T]] = Vec(numRdWr, new MemoryReadWritePort(tpe, addrWidth, masked))
}

object MemInterface {

  /** Generates a [[SyncReadMem]] connected to an explicit number of read, write,
    * and read/write ports within the current module.
    *
    * @param size The desired size of the inner `SyncReadMem`
    * @tparam T The data type of the memory element
    * @param numRd The number of desired read ports, >= 0
    * @param numWr The number of desired write ports, >= 0
    * @param numRdWr The number of desired read/write ports, >= 0
    *
    * @return A new `MemInterface` wire containing the control signals for each instantiated port
    * @note This does *not* return the `SyncReadMem` itself, you must interact with it using the returned bundle
    */
  def apply[T <: Data](
    size:    BigInt,
    tpe:     T,
    numRd:   Int,
    numWr:   Int,
    numRdWr: Int
  ): MemInterface[T] =
    macro MemTransform.apply_memInterface[T]

  /** Generates a [[SyncReadMem]] wrapper connected to an explicit number of read, write,
    * and read/write ports, with masking capability on all write and read/write ports,
    * within the current module.
    *
    * @param size The desired size of the inner `SyncReadMem`
    * @tparam T The data type of the memory element
    * @param numRd The number of desired read ports, >= 0
    * @param numWr The number of desired write ports, >= 0
    * @param numRdWr The number of desired read/write ports, >= 0
    *
    * @return A new `MemInterface` wire containing the control signals for each instantiated port
    * @note This does *not* return the `SyncReadMem` itself, you must interact with it using the returned bundle
    */
  def masked[T <: Data](
    size:    BigInt,
    tpe:     T,
    numRd:   Int,
    numWr:   Int,
    numRdWr: Int
  )(
    implicit evidence: T <:< Vec[_]
  ): MemInterface[T] =
    macro MemTransform.masked_memInterface[T]

  /** @group SourceInfoTransformMacro */
  def do_apply[T <: Data](
    size:    BigInt,
    tpe:     T,
    numRd:   Int,
    numWr:   Int,
    numRdWr: Int
  )(
    implicit sourceInfo: SourceInfo
  ): MemInterface[T] = memInterface_impl(size, tpe)(numRd, numWr, numRdWr, Builder.forcedClock)

  /** @group SourceInfoTransformMacro */
  def do_masked[T <: Data](
    size:    BigInt,
    tpe:     T,
    numRd:   Int,
    numWr:   Int,
    numRdWr: Int
  )(
    implicit sourceInfo: SourceInfo,
    evidence:            T <:< Vec[_]
  ): MemInterface[T] = masked_memInterface_impl(size, tpe)(numRd, numWr, numRdWr, Builder.forcedClock)

  /** @group SourceInfoTransformMacro */
  private def memInterface_impl[T <: Data](
    size:    BigInt,
    tpe:     T
  )(numRd:   Int,
    numWr:   Int,
    numRdWr: Int,
    clock:   Clock
  )(
    implicit sourceInfo: SourceInfo
  ): MemInterface[T] = {
    val addrWidth = log2Up(size + 1)
    
    val _out = Wire(new MemInterface(tpe, addrWidth, numRd, numWr, numRdWr, false))
    val mem = SyncReadMem(size, tpe)

    for (i <- 0 until numRd) {
      _out.rd(i).readValue := mem.read(_out.rd(i).addr, _out.rd(i).enable, clock)
    }

    for (i <- 0 until numWr) {
      when(_out.wr(i).enable) {
        mem.write(_out.wr(i).addr, _out.wr(i).writeValue, clock)
      }
    }

    for (i <- 0 until numRdWr) {
      _out.rw(i).readValue := mem.readWrite(
        _out.rw(i).addr,
        _out.rw(i).writeValue,
        _out.rw(i).enable,
        _out.rw(i).isWrite,
        clock
      )
    }
    _out
  }

  /** @group SourceInfoTransformMacro */
  private def masked_memInterface_impl[T <: Data](
    size:    BigInt,
    tpe:     T
  )(numRd:   Int,
    numWr:   Int,
    numRdWr: Int,
    clock:   Clock
  )(
    implicit sourceInfo: SourceInfo,
    evidence:            T <:< Vec[_]
  ): MemInterface[T] = {
    val addrWidth = log2Up(size + 1)

    val _out = Wire(new MemInterface(tpe, addrWidth, numRd, numWr, numRdWr, true))
    val mem = SyncReadMem(size, tpe)

    for (i <- 0 until numRd) {
      _out.rd(i).readValue := mem.read(_out.rd(i).addr, _out.rd(i).enable, clock)
    }

    for (i <- 0 until numWr) {
      when(_out.wr(i).enable) {
        mem.write(_out.wr(i).addr, _out.wr(i).writeValue, _out.wr(i).mask.get, clock)
      }
    }

    for (i <- 0 until numRdWr) {
      _out.rw(i).readValue := mem.readWrite(
        _out.rw(i).addr,
        _out.rw(i).writeValue,
        _out.rw(i).mask.get,
        _out.rw(i).enable,
        _out.rw(i).isWrite,
        clock
      )
    }

    _out
  }

  // Helper util to generate portedness descriptors based on the input parameters
  // supplied to MemInterface.apply
  def portedness(rd: Int, wr: Int, rw: Int): String = {
    val rdPorts: String = if (rd > 0) s"${rd}R" else ""
    val wrPorts: String = if (wr > 0) s"${wr}W" else ""
    val rwPorts: String = if (rw > 0) s"${rw}RW" else ""

    s"$rdPorts$wrPorts$rwPorts"
  }
}
