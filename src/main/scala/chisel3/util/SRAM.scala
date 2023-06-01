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
  val address = Input(UInt(addrWidth.W))
  val enable = Input(Bool())
  val data = Output(tpe)
  val clock = Input(Clock())
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
  val address = Input(UInt(addrWidth.W))
  val enable = Input(Bool())
  val data = Input(tpe)
  val mask: Option[Vec[Bool]] = if (masked) {
    val maskSize = tpe match {
      case vec: Vec[_] => vec.size
      case _ => 0
    }
    Some(Input(Vec(maskSize, Bool())))
  } else {
    None
  }
  val clock = Input(Clock())
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
  val address = Input(UInt(addrWidth.W))
  val enable = Input(Bool())
  val isWrite = Input(Bool())
  val readData = Output(tpe)
  val writeData = Input(tpe)
  val mask: Option[Vec[Bool]] = if (masked) {
    val maskSize = tpe match {
      case vec: Vec[_] => vec.size
      case _ => 0
    }
    Some(Input(Vec(maskSize, Bool())))
  } else {
    None
  }
  val clock = Input(Clock())
}

/** A IO bundle of signals connecting to the ports of a wrapped `SyncReadMem`, as requested by
  * `SRAMInterface.apply`.
  *
  * @tparam tpe The data type of the memory port
  * @param width The width of the address wires of each port
  * @param numReadPorts The number of read ports
  * @param numWritePorts The number of write ports
  * @param numReadwritePorts The number of read/write ports
  */
class SRAMInterface[T <: Data](
  tpe:               T,
  addrWidth:         Int,
  numReadPorts:      Int,
  numWritePorts:     Int,
  numReadwritePorts: Int,
  masked:            Boolean)
    extends Bundle {
  if (masked) {
    require(
      tpe.isInstanceOf[Vec[_]],
      s"masked writes require that SRAMInterface is instantiated with a data type of Vec (got $tpe instead)"
    )
  }
  override def typeName: String = s"SRAMInterface_${SRAM.portedness(numReadPorts, numWritePorts, numReadwritePorts)}"

  val readPorts:  Vec[MemoryReadPort[T]] = Vec(numReadPorts, new MemoryReadPort(tpe, addrWidth))
  val writePorts: Vec[MemoryWritePort[T]] = Vec(numWritePorts, new MemoryWritePort(tpe, addrWidth, masked))
  val readwritePorts: Vec[MemoryReadWritePort[T]] =
    Vec(numReadwritePorts, new MemoryReadWritePort(tpe, addrWidth, masked))
}

object SRAM {

  /** Generates a [[SyncReadMem]] within the current module, connected to an explicit number
    * of read, write, and read/write ports
    *
    * @param size The desired size of the inner `SyncReadMem`
    * @tparam T The data type of the memory element
    * @param numReadPorts The number of desired read ports, >= 0
    * @param numWritePorts The number of desired write ports, >= 0
    * @param numReadwritePorts The number of desired read/write ports, >= 0
    *
    * @return A new `SRAMInterface` wire containing the control signals for each instantiated port
    * @note This does *not* return the `SyncReadMem` itself, you must interact with it using the returned bundle
    */
  def apply[T <: Data](
    size:              BigInt,
    tpe:               T,
    numReadPorts:      Int,
    numWritePorts:     Int,
    numReadwritePorts: Int
  )(
    implicit sourceInfo: SourceInfo
  ): SRAMInterface[T] =
    memInterface_impl(size, tpe)(numReadPorts, numWritePorts, numReadwritePorts, Builder.forcedClock)

  /** Generates a [[SyncReadMem]] within the current module, connected to an explicit number
    * of read, write, and read/write ports, with masking capability on all write and read/write ports.
    *
    * @param size The desired size of the inner `SyncReadMem`
    * @tparam T The data type of the memory element
    * @param numReadPorts The number of desired read ports, >= 0
    * @param numWritePorts The number of desired write ports, >= 0
    * @param numReadwritePorts The number of desired read/write ports, >= 0
    *
    * @return A new `SRAMInterface` wire containing the control signals for each instantiated port
    * @note This does *not* return the `SyncReadMem` itself, you must interact with it using the returned bundle
    */
  def masked[T <: Data](
    size:              BigInt,
    tpe:               T,
    numReadPorts:      Int,
    numWritePorts:     Int,
    numReadwritePorts: Int
  )(
    implicit evidence: T <:< Vec[_],
    sourceInfo:        SourceInfo
  ): SRAMInterface[T] =
    memInterface_impl(size, tpe)(numReadPorts, numWritePorts, numReadwritePorts, Builder.forcedClock)

  /** @group SourceInfoTransformMacro */
  private def memInterface_impl[T <: Data](
    size:              BigInt,
    tpe:               T
  )(numReadPorts:      Int,
    numWritePorts:     Int,
    numReadwritePorts: Int,
    clock:             Clock
  )(
    implicit sourceInfo: SourceInfo
  ): SRAMInterface[T] = {
    val addrWidth = log2Up(size + 1)

    val _out = Wire(new SRAMInterface(tpe, addrWidth, numReadPorts, numWritePorts, numReadwritePorts, false))
    val mem = SyncReadMem(size, tpe)

    for (i <- 0 until numReadPorts) {
      _out.readPorts(i).data := mem.read(_out.readPorts(i).address, _out.readPorts(i).enable, _out.readPorts(i).clock)
      _out.readPorts(i).clock :#= clock
    }

    for (i <- 0 until numWritePorts) {
      when(_out.writePorts(i).enable) {
        mem.write(_out.writePorts(i).address, _out.writePorts(i).data, _out.writePorts(i).clock)
      }
      _out.writePorts(i).clock :#= clock
    }

    for (i <- 0 until numReadwritePorts) {
      _out.readwritePorts(i).readData := mem.readWrite(
        _out.readwritePorts(i).address,
        _out.readwritePorts(i).writeData,
        _out.readwritePorts(i).enable,
        _out.readwritePorts(i).isWrite,
        _out.readwritePorts(i).clock
      )
      _out.readwritePorts(i).clock :#= clock
    }

    _out
  }

  /** @group SourceInfoTransformMacro */
  private def masked_memInterface_impl[T <: Data](
    size:              BigInt,
    tpe:               T
  )(numReadPorts:      Int,
    numWritePorts:     Int,
    numReadwritePorts: Int,
    clock:             Clock
  )(
    implicit sourceInfo: SourceInfo,
    evidence:            T <:< Vec[_]
  ): SRAMInterface[T] = {
    val addrWidth = log2Up(size + 1)

    val _out = Wire(new SRAMInterface(tpe, addrWidth, numReadPorts, numWritePorts, numReadwritePorts, true))
    val mem = SyncReadMem(size, tpe)

    for (i <- 0 until numReadPorts) {
      _out.readPorts(i).clock :#= clock
      _out.readPorts(i).data := mem.read(_out.readPorts(i).address, _out.readPorts(i).enable, _out.readPorts(i).clock)
    }

    for (i <- 0 until numWritePorts) {
      _out.writePorts(i).clock :#= clock
      when(_out.writePorts(i).enable) {
        mem.write(
          _out.writePorts(i).address,
          _out.writePorts(i).data,
          _out.writePorts(i).mask.get,
          _out.writePorts(i).clock
        )
      }
    }

    for (i <- 0 until numReadwritePorts) {
      _out.readwritePorts(i).clock := clock
      _out.readwritePorts(i).readData := mem.readWrite(
        _out.readwritePorts(i).address,
        _out.readwritePorts(i).writeData,
        _out.readwritePorts(i).mask.get,
        _out.readwritePorts(i).enable,
        _out.readwritePorts(i).isWrite,
        _out.readwritePorts(i).clock
      )
    }

    _out
  }

  // Helper util to generate portedness descriptors based on the input parameters
  // supplied to SRAM.apply
  private[chisel3] def portedness(rd: Int, wr: Int, rw: Int): String = {
    val rdPorts: String = if (rd > 0) s"${rd}R" else ""
    val wrPorts: String = if (wr > 0) s"${wr}W" else ""
    val rwPorts: String = if (rw > 0) s"${rw}RW" else ""

    s"$rdPorts$wrPorts$rwPorts"
  }
}
