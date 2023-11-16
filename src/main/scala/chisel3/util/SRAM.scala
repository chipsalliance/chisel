package chisel3.util

import chisel3._

import chisel3.internal.Builder
import chisel3.experimental.SourceInfo
import chisel3.internal.sourceinfo.{MemTransform, SourceInfoTransform}
import chisel3.util.experimental.loadMemoryFromFileInline
import firrtl.annotations.MemoryLoadFileType
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
}

/** A IO bundle of signals connecting to the ports of a wrapped `SyncReadMem`, as requested by
  * `SRAMInterface.apply`.
  *
  * @param memSize The size of the memory, used to calculate the address width
  * @tparam tpe The data type of the memory port
  * @param numReadPorts The number of read ports
  * @param numWritePorts The number of write ports
  * @param numReadwritePorts The number of read/write ports
  */
class SRAMInterface[T <: Data](
  memSize:           BigInt,
  tpe:               T,
  numReadPorts:      Int,
  numWritePorts:     Int,
  numReadwritePorts: Int,
  masked:            Boolean = false)
    extends Bundle {
  if (masked) {
    require(
      tpe.isInstanceOf[Vec[_]],
      s"masked writes require that SRAMInterface is instantiated with a data type of Vec (got $tpe instead)"
    )
  }
  override def typeName: String =
    s"SRAMInterface_${SRAM.portedness(numReadPorts, numWritePorts, numReadwritePorts)}${if (masked) "_masked"
    else ""}}_${tpe.typeName}"

  val addrWidth = log2Up(memSize + 1)

  val readPorts:  Vec[MemoryReadPort[T]] = Vec(numReadPorts, new MemoryReadPort(tpe, addrWidth))
  val writePorts: Vec[MemoryWritePort[T]] = Vec(numWritePorts, new MemoryWritePort(tpe, addrWidth, masked))
  val readwritePorts: Vec[MemoryReadWritePort[T]] =
    Vec(numReadwritePorts, new MemoryReadWritePort(tpe, addrWidth, masked))
}

/** A memory file with which to preload an [[SRAM]]
  *
  * See concrete subclasses [[BinaryMemoryFile]] and [[HexMemoryFile]]
  */
sealed abstract class MemoryFile(private[chisel3] val fileType: MemoryLoadFileType) {

  /** The path to the memory contents file */
  val path: String
}

/** A binary memory file to preload an [[SRAM]] with, represented by a filesystem path. This will annotate
  * the inner [[SyncReadMem]] with `loadMemoryFromFile` using `MemoryLoadFileType.Binary` as the file type.
  *
  * @param path The path to the binary file
  */
case class BinaryMemoryFile(path: String) extends MemoryFile(MemoryLoadFileType.Binary)

/** A hex memory file to preload an [[SRAM]] with, represented by a filesystem path. This will annotate
  * the inner [[SyncReadMem]] with `loadMemoryFromFile` using `MemoryLoadFileType.Hex` as the file type.
  *
  * @param path The path to the hex file
  */
case class HexMemoryFile(path: String) extends MemoryFile(MemoryLoadFileType.Hex)

object SRAM {

  /** Generates a [[SyncReadMem]] within the current module, connected to an explicit number
    * of read, write, and read/write ports. This SRAM abstraction has both read and write capabilities: that is,
    * it contains at least one read accessor (a read-only or read-write port), and at least one write accessor
    * (a write-only or read-write port).
    *
    * @param size The desired size of the inner `SyncReadMem`
    * @tparam T The data type of the memory element
    * @param numReadPorts The number of desired read ports >= 0, and (numReadPorts + numReadwritePorts) > 0
    * @param numWritePorts The number of desired write ports >= 0, and (numWritePorts + numReadwritePorts) > 0
    * @param numReadwritePorts The number of desired read/write ports >= 0, and the above two conditions must hold
    *
    * @return A new `SRAMInterface` wire containing the control signals for each instantiated port
    * @note This does *not* return the `SyncReadMem` itself, you must interact with it using the returned bundle
    * @note Read-only memories (R >= 1, W === 0, RW === 0) and write-only memories (R === 0, W >= 1, RW === 0) are not supported by this API, and will result in an error if declared.
    */
  def apply[T <: Data](
    size:              BigInt,
    tpe:               T,
    numReadPorts:      Int,
    numWritePorts:     Int,
    numReadwritePorts: Int
  )(
    implicit sourceInfo: SourceInfo
  ): SRAMInterface[T] = {
    val clock = Builder.forcedClock
    memInterface_impl(
      size,
      tpe,
      Seq.fill(numReadPorts)(clock),
      Seq.fill(numWritePorts)(clock),
      Seq.fill(numReadwritePorts)(clock),
      None,
      None,
      sourceInfo
    )
  }

  /** Generates a [[SyncReadMem]] within the current module, connected to an explicit number
    * of read, write, and read/write ports. This SRAM abstraction has both read and write capabilities: that is,
    * it contains at least one read accessor (a read-only or read-write port), and at least one write accessor
    * (a write-only or read-write port).
    *
    * @param size The desired size of the inner `SyncReadMem`
    * @tparam T The data type of the memory element
    * @param numReadPorts The number of desired read ports >= 0, and (numReadPorts + numReadwritePorts) > 0
    * @param numWritePorts The number of desired write ports >= 0, and (numWritePorts + numReadwritePorts) > 0
    * @param numReadwritePorts The number of desired read/write ports >= 0, and the above two conditions must hold
    * @param memoryFile A memory file whose path is emitted as Verilog directives to initialize the inner `SyncReadMem`
    *
    * @return A new `SRAMInterface` wire containing the control signals for each instantiated port
    * @note This does *not* return the `SyncReadMem` itself, you must interact with it using the returned bundle
    * @note Read-only memories (R >= 1, W === 0, RW === 0) and write-only memories (R === 0, W >= 1, RW === 0) are not supported by this API, and will result in an error if declared.
    */
  def apply[T <: Data](
    size:              BigInt,
    tpe:               T,
    numReadPorts:      Int,
    numWritePorts:     Int,
    numReadwritePorts: Int,
    memoryFile:        MemoryFile
  )(
    implicit sourceInfo: SourceInfo
  ): SRAMInterface[T] = {
    val clock = Builder.forcedClock
    memInterface_impl(
      size,
      tpe,
      Seq.fill(numReadPorts)(clock),
      Seq.fill(numWritePorts)(clock),
      Seq.fill(numReadwritePorts)(clock),
      Some(memoryFile),
      None,
      sourceInfo
    )
  }

  /** Generates a [[SyncReadMem]] within the current module, connected to an explicit number
    * of read, write, and read/write ports. This SRAM abstraction has both read and write capabilities: that is,
    * it contains at least one read accessor (a read-only or read-write port), and at least one write accessor
    * (a write-only or read-write port).
    *
    * @param size The desired size of the inner `SyncReadMem`
    * @tparam T The data type of the memory element
    * @param numReadPorts The number of desired read ports >= 0, and (numReadPorts + numReadwritePorts) > 0
    * @param numWritePorts The number of desired write ports >= 0, and (numWritePorts + numReadwritePorts) > 0
    * @param numReadwritePorts The number of desired read/write ports >= 0, and the above two conditions must hold
    *
    * @return A new `SRAMInterface` wire containing the control signals for each instantiated port
    * @note This does *not* return the `SyncReadMem` itself, you must interact with it using the returned bundle
    * @note Read-only memories (R >= 1, W === 0, RW === 0) and write-only memories (R === 0, W >= 1, RW === 0) are not supported by this API, and will result in an error if declared.
    */
  def apply[T <: Data](
    size:                BigInt,
    tpe:                 T,
    readPortClocks:      Seq[Clock],
    writePortClocks:     Seq[Clock],
    readwritePortClocks: Seq[Clock]
  )(
    implicit sourceInfo: SourceInfo
  ): SRAMInterface[T] =
    memInterface_impl(
      size,
      tpe,
      readPortClocks,
      writePortClocks,
      readwritePortClocks,
      None,
      None,
      sourceInfo
    )

  /** Generates a [[SyncReadMem]] within the current module, connected to an explicit number
    * of read, write, and read/write ports. This SRAM abstraction has both read and write capabilities: that is,
    * it contains at least one read accessor (a read-only or read-write port), and at least one write accessor
    * (a write-only or read-write port).
    *
    * @param size The desired size of the inner `SyncReadMem`
    * @tparam T The data type of the memory element
    * @param numReadPorts The number of desired read ports >= 0, and (numReadPorts + numReadwritePorts) > 0
    * @param numWritePorts The number of desired write ports >= 0, and (numWritePorts + numReadwritePorts) > 0
    * @param numReadwritePorts The number of desired read/write ports >= 0, and the above two conditions must hold
    * @param memoryFile A memory file whose path is emitted as Verilog directives to initialize the inner `SyncReadMem`
    *
    * @return A new `SRAMInterface` wire containing the control signals for each instantiated port
    * @note This does *not* return the `SyncReadMem` itself, you must interact with it using the returned bundle
    * @note Read-only memories (R >= 1, W === 0, RW === 0) and write-only memories (R === 0, W >= 1, RW === 0) are not supported by this API, and will result in an error if declared.
    */
  def apply[T <: Data](
    size:                BigInt,
    tpe:                 T,
    readPortClocks:      Seq[Clock],
    writePortClocks:     Seq[Clock],
    readwritePortClocks: Seq[Clock],
    memoryFile:          MemoryFile
  )(
    implicit sourceInfo: SourceInfo
  ): SRAMInterface[T] =
    memInterface_impl(
      size,
      tpe,
      readPortClocks,
      writePortClocks,
      readwritePortClocks,
      Some(memoryFile),
      None,
      sourceInfo
    )

  /** Generates a [[SyncReadMem]] within the current module, connected to an explicit number
    * of read, write, and read/write ports, with masking capability on all write and read/write ports.
    * This SRAM abstraction has both read and write capabilities: that is, it contains at least one read
    * accessor (a read-only or read-write port), and at least one write accessor (a write-only or read-write port).
    *
    * @param size The desired size of the inner `SyncReadMem`
    * @tparam T The data type of the memory element
    * @param numReadPorts The number of desired read ports >= 0, and (numReadPorts + numReadwritePorts) > 0
    * @param numWritePorts The number of desired write ports >= 0, and (numWritePorts + numReadwritePorts) > 0
    * @param numReadwritePorts The number of desired read/write ports >= 0, and the above two conditions must hold
    *
    * @return A new `SRAMInterface` wire containing the control signals for each instantiated port
    * @note This does *not* return the `SyncReadMem` itself, you must interact with it using the returned bundle
    * @note Read-only memories (R >= 1, W === 0, RW === 0) and write-only memories (R === 0, W >= 1, RW === 0) are not supported by this API, and will result in an error if declared.
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
  ): SRAMInterface[T] = {
    val clock = Builder.forcedClock
    memInterface_impl(
      size,
      tpe,
      Seq.fill(numReadPorts)(clock),
      Seq.fill(numWritePorts)(clock),
      Seq.fill(numReadwritePorts)(clock),
      None,
      Some(evidence),
      sourceInfo
    )
  }

  /** Generates a [[SyncReadMem]] within the current module, connected to an explicit number
    * of read, write, and read/write ports, with masking capability on all write and read/write ports.
    * This SRAM abstraction has both read and write capabilities: that is, it contains at least one read
    * accessor (a read-only or read-write port), and at least one write accessor (a write-only or read-write port).
    *
    * @param size The desired size of the inner `SyncReadMem`
    * @tparam T The data type of the memory element
    * @param numReadPorts The number of desired read ports >= 0, and (numReadPorts + numReadwritePorts) > 0
    * @param numWritePorts The number of desired write ports >= 0, and (numWritePorts + numReadwritePorts) > 0
    * @param numReadwritePorts The number of desired read/write ports >= 0, and the above two conditions must hold
    * @param memoryFile A memory file whose path is emitted as Verilog directives to initialize the inner `SyncReadMem`
    *
    * @return A new `SRAMInterface` wire containing the control signals for each instantiated port
    * @note This does *not* return the `SyncReadMem` itself, you must interact with it using the returned bundle
    * @note Read-only memories (R >= 1, W === 0, RW === 0) and write-only memories (R === 0, W >= 1, RW === 0) are not supported by this API, and will result in an error if declared.
    */
  def masked[T <: Data](
    size:              BigInt,
    tpe:               T,
    numReadPorts:      Int,
    numWritePorts:     Int,
    numReadwritePorts: Int,
    memoryFile:        MemoryFile
  )(
    implicit evidence: T <:< Vec[_],
    sourceInfo:        SourceInfo
  ): SRAMInterface[T] = {
    val clock = Builder.forcedClock
    memInterface_impl(
      size,
      tpe,
      Seq.fill(numReadPorts)(clock),
      Seq.fill(numWritePorts)(clock),
      Seq.fill(numReadwritePorts)(clock),
      Some(memoryFile),
      Some(evidence),
      sourceInfo
    )
  }

  /** Generates a [[SyncReadMem]] within the current module, connected to an explicit number
    * of read, write, and read/write ports, with masking capability on all write and read/write ports.
    * Each port is clocked with its own explicit `Clock`, rather than being given the implicit clock.
    *
    * @param size The desired size of the inner `SyncReadMem`
    * @tparam T The data type of the memory element
    * @param readPortClocks A sequence of clocks for each read port; and (numReadPorts + numReadwritePorts) > 0
    * @param writePortClocks A sequence of clocks for each write port; and (numWritePorts + numReadwritePorts) > 0
    * @param readwritePortClocks A sequence of clocks for each read-write port; and the above two conditions must hold
    *
    * @return A new `SRAMInterface` wire containing the control signals for each instantiated port
    * @note The size of each `Clock` sequence determines the corresponding number of read, write, and read-write ports
    * @note This does *not* return the `SyncReadMem` itself, you must interact with it using the returned bundle
    * @note Read-only memories (R >= 1, W === 0, RW === 0) and write-only memories (R === 0, W >= 1, RW === 0) are not supported by this API, and will result in an error if declared.
    */
  def masked[T <: Data](
    size:                BigInt,
    tpe:                 T,
    readPortClocks:      Seq[Clock],
    writePortClocks:     Seq[Clock],
    readwritePortClocks: Seq[Clock]
  )(
    implicit evidence: T <:< Vec[_],
    sourceInfo:        SourceInfo
  ): SRAMInterface[T] =
    memInterface_impl(
      size,
      tpe,
      readPortClocks,
      writePortClocks,
      readwritePortClocks,
      None,
      Some(evidence),
      sourceInfo
    )

  /** Generates a [[SyncReadMem]] within the current module, connected to an explicit number
    * of read, write, and read/write ports, with masking capability on all write and read/write ports.
    * Each port is clocked with its own explicit `Clock`, rather than being given the implicit clock.
    *
    * @param size The desired size of the inner `SyncReadMem`
    * @tparam T The data type of the memory element
    * @param readPortClocks A sequence of clocks for each read port; and (numReadPorts + numReadwritePorts) > 0
    * @param writePortClocks A sequence of clocks for each write port; and (numWritePorts + numReadwritePorts) > 0
    * @param readwritePortClocks A sequence of clocks for each read-write port; and the above two conditions must hold
    * @param memoryFile A memory file whose path is emitted as Verilog directives to initialize the inner `SyncReadMem`
    *
    * @return A new `SRAMInterface` wire containing the control signals for each instantiated port
    * @note The size of each `Clock` sequence determines the corresponding number of read, write, and read-write ports
    * @note This does *not* return the `SyncReadMem` itself, you must interact with it using the returned bundle
    * @note Read-only memories (R >= 1, W === 0, RW === 0) and write-only memories (R === 0, W >= 1, RW === 0) are not supported by this API, and will result in an error if declared.
    */
  def masked[T <: Data](
    size:                BigInt,
    tpe:                 T,
    readPortClocks:      Seq[Clock],
    writePortClocks:     Seq[Clock],
    readwritePortClocks: Seq[Clock],
    memoryFile:          MemoryFile
  )(
    implicit evidence: T <:< Vec[_],
    sourceInfo:        SourceInfo
  ): SRAMInterface[T] =
    memInterface_impl(
      size,
      tpe,
      readPortClocks,
      writePortClocks,
      readwritePortClocks,
      Some(memoryFile),
      Some(evidence),
      sourceInfo
    )

  private def memInterface_impl[T <: Data](
    size:                BigInt,
    tpe:                 T,
    readPortClocks:      Seq[Clock],
    writePortClocks:     Seq[Clock],
    readwritePortClocks: Seq[Clock],
    memoryFile:          Option[MemoryFile],
    evidenceOpt:         Option[T <:< Vec[_]],
    sourceInfo:          SourceInfo
  ): SRAMInterface[T] = {
    val numReadPorts = readPortClocks.size
    val numWritePorts = writePortClocks.size
    val numReadwritePorts = readwritePortClocks.size
    val isVecMem = evidenceOpt.isDefined
    val isValidSRAM = ((numReadPorts + numReadwritePorts) > 0) && ((numWritePorts + numReadwritePorts) > 0)

    if (!isValidSRAM) {
      val badMemory =
        if (numReadPorts + numReadwritePorts == 0)
          "write-only SRAM (R + RW === 0)"
        else
          "read-only SRAM (W + RW === 0)"
      Builder.error(
        s"Attempted to initialize a $badMemory! SRAMs must have both at least one read accessor and at least one write accessor."
      )
    }

    val _out = Wire(new SRAMInterface(size, tpe, numReadPorts, numWritePorts, numReadwritePorts, isVecMem))
    val mem = SyncReadMem(size, tpe)

    for ((clock, port) <- readPortClocks.zip(_out.readPorts)) {
      port.data := mem.read(port.address, port.enable, clock)
    }

    for ((clock, port) <- writePortClocks.zip(_out.writePorts)) {
      when(port.enable) {
        if (isVecMem) {
          mem.write(
            port.address,
            port.data,
            port.mask.get,
            clock
          )(evidenceOpt.get)
        } else {
          mem.write(port.address, port.data, clock)
        }
      }
    }

    for ((clock, port) <- readwritePortClocks.zip(_out.readwritePorts)) {
      if (isVecMem) {
        port.readData := mem.readWrite(
          port.address,
          port.writeData,
          port.mask.get,
          port.enable,
          port.isWrite,
          clock
        )(evidenceOpt.get)
      } else {
        port.readData := mem.readWrite(
          port.address,
          port.writeData,
          port.enable,
          port.isWrite,
          clock
        )
      }
    }

    // Emit Verilog for preloading the memory from a file if requested
    memoryFile.foreach { file: MemoryFile => loadMemoryFromFileInline(mem, file.path, file.fileType) }

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
