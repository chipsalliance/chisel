package chisel3.util

import chisel3._

import chisel3.internal.Builder
import chisel3.experimental.SourceInfo
import chisel3.internal.sourceinfo.{MemTransform, SourceInfoTransform}
import scala.language.reflectiveCalls
import scala.language.experimental.macros

/** A bundle of signals representing a read memory port.
  *
  * @tparam tpe The data type of the memory port
  * @param addrWidth The width of the address signal
  */
class MemRdPortInterface[T <: Data](tpe: T, addrWidth: Int) extends Bundle {
  val addr = Input(UInt(addrWidth.W))
  val enable = Input(Bool())
  val readValue = Output(tpe)
}

/** A bundle of signals representing a write memory port.
  *
  * @tparam tpe The data type of the memory port
  * @param addrWidth The width of the address signal
  */
class MemWrPortInterface[T <: Data](tpe: T, addrWidth: Int) extends Bundle {
  val addr = Input(UInt(addrWidth.W))
  val enable = Input(Bool())
  val writeValue = Input(tpe)
}

/** A bundle of signals representing a read/write memory port.
  *
  * @tparam tpe The data type of the memory port
  * @param addrWidth The width of the address signal
  */
class MemRdWrPortInterface[T <: Data](tpe: T, addrWidth: Int) extends Bundle {
  val addr = Input(UInt(addrWidth.W))
  val enable = Input(Bool())
  val isWrite = Input(Bool())
  val readValue = Output(tpe)
  val writeValue = Input(tpe)
}

/** A IO bundle of signals connecting to the ports of a wrapped `SyncReadMem`, as requested by
  * `SyncReadMem.interface`.
  *
  * @tparam tpe The data type of the memory port
  * @param width The width of the address wires of each port
  * @param numRd The number of read ports
  * @param numWr The number of write ports
  * @param numRdWr The number of read/write ports
  */
class MemInterface[T <: Data](tpe: T, addrWidth: Int, numRd: Int, numWr: Int, numRdWr: Int) extends Bundle {
  override def typeName: String = s"MemInterface_${MemInterface.portedness(numRd, numWr, numRdWr)}"

  val rd: Vec[MemRdPortInterface[T]] = Vec(numRd, new MemRdPortInterface(tpe, addrWidth))
  val wr: Vec[MemWrPortInterface[T]] = Vec(numWr, new MemWrPortInterface(tpe, addrWidth))
  val rw: Vec[MemRdWrPortInterface[T]] = Vec(numRdWr, new MemRdWrPortInterface(tpe, addrWidth))
}

object MemInterface {

  /** Generates a [[SyncReadMem]] wrapper connected to an explicit number of read, write,
    * and read/write ports
    *
    * @param size The desired size of the inner `SyncReadMem`
    * @tparam T The data type of the memory element
    * @param numRd The number of desired read ports, >= 0
    * @param numWr The number of desired write ports, >= 0
    * @param numRdWr The number of desired read/write ports, >= 0
    *
    * @return A new `MemInterface` wire containing the control signals for each instantiated port
    * @note This does *not* return the wrapper module itself, you must interact with it using the returned bundle
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
    * and read/write ports
    *
    * @param size The desired size of the inner `SyncReadMem`
    * @tparam T The data type of the memory element
    * @param numRd The number of desired read ports, >= 0
    * @param numWr The number of desired write ports, >= 0
    * @param numRdWr The number of desired read/write ports, >= 0
    * @param clock The clock to bind to each generated port, which may be different from the implicit clock
    *
    * @return A new `MemInterface` wire containing the control signals for each instantiated port
    * @note This does *not* return the wrapper module itself, you must interact with it using the returned bundle
    */
  def apply[T <: Data](
    size:    BigInt,
    tpe:     T,
    numRd:   Int,
    numWr:   Int,
    numRdWr: Int,
    clock:   Clock
  ): MemInterface[T] =
    macro MemTransform.apply_memInterfaceClk[T]

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
  def do_apply[T <: Data](
    size:    BigInt,
    tpe:     T,
    numRd:   Int,
    numWr:   Int,
    numRdWr: Int,
    clock:   Clock
  )(
    implicit sourceInfo: SourceInfo
  ): MemInterface[T] = memInterface_impl(size, tpe)(numRd, numWr, numRdWr, clock)

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

    val _wrappedMem = Module(new Module {
      override def desiredName: String =
        s"SyncReadMemWrapper_${MemInterface.portedness(numRd, numWr, numRdWr)}_${tpe.typeName}"

      val io = IO(new MemInterface(tpe, addrWidth, numRd, numWr, numRdWr))

      val _innerMem = SyncReadMem(size, tpe)

      for (i <- 0 until numRd) {
        io.rd(i).readValue := _innerMem.read(io.rd(i).addr, io.rd(i).enable, clock)
      }

      for (i <- 0 until numWr) {
        when(io.wr(i).enable) {
          _innerMem.write(io.wr(i).addr, io.wr(i).writeValue, clock)
        }
      }

      for (i <- 0 until numRdWr) {
        io.rw(i).readValue := _innerMem.readWrite(
          io.rw(i).addr,
          io.rw(i).writeValue,
          io.rw(i).enable,
          io.rw(i).isWrite,
          clock
        )
      }
    })

    val _out = Wire(new MemInterface(tpe, addrWidth, numRd, numWr, numRdWr))
    _wrappedMem.io <> _out
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
