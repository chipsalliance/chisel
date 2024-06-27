package chisel3.util

import chisel3._

import chisel3.internal.{Builder, NamedComponent}
import chisel3.internal.binding.{FirrtlMemTypeBinding, SramPortBinding}
import chisel3.internal.plugin.autoNameRecursively
import chisel3.experimental.SourceInfo
import chisel3.internal.sourceinfo.{MemTransform, SourceInfoTransform}
import chisel3.internal.firrtl.ir.{Arg, FirrtlMemory, ILit, Index, Node, Ref, Slot}
import chisel3.util.experimental.loadMemoryFromFileInline
import firrtl.annotations.MemoryLoadFileType
import scala.language.reflectiveCalls
import scala.language.experimental.macros
import chisel3.internal.firrtl.ir

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
class MemoryWritePort[T <: Data](tpe: T, addrWidth: Int, masked: Boolean) extends Bundle {
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
class MemoryReadWritePort[T <: Data](tpe: T, addrWidth: Int, masked: Boolean) extends Bundle {
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

/** A IO bundle of signals connecting to the ports of a memory, as requested by
  * `SRAMInterface.apply`.
  *
  * @param memSize The size of the memory, used to calculate the address width
  * @tparam tpe The data type of the memory port
  * @param numReadPorts The number of read ports
  * @param numWritePorts The number of write ports
  * @param numReadwritePorts The number of read/write ports
  * @param masked Whether the memory is write masked
  */
class SRAMInterface[T <: Data](
  val memSize: BigInt,
  // tpe can't be directly made public as it will become a Bundle field
  tpe:                   T,
  val numReadPorts:      Int,
  val numWritePorts:     Int,
  val numReadwritePorts: Int,
  val masked:            Boolean = false)
    extends Bundle {

  /** Public accessor for data type of this interface. */
  def dataType: T = tpe

  if (masked) {
    require(
      tpe.isInstanceOf[Vec[_]],
      s"masked writes require that SRAMInterface is instantiated with a data type of Vec (got $tpe instead)"
    )
  }
  override def typeName: String =
    s"SRAMInterface_${SRAM.portedness(numReadPorts, numWritePorts, numReadwritePorts)}${if (masked) "_masked"
    else ""}}_${tpe.typeName}"

  val addrWidth = log2Up(memSize)

  val readPorts:  Vec[MemoryReadPort[T]] = Vec(numReadPorts, new MemoryReadPort(tpe, addrWidth))
  val writePorts: Vec[MemoryWritePort[T]] = Vec(numWritePorts, new MemoryWritePort(tpe, addrWidth, masked))
  val readwritePorts: Vec[MemoryReadWritePort[T]] =
    Vec(numReadwritePorts, new MemoryReadWritePort(tpe, addrWidth, masked))

  private[chisel3] var _underlying: Option[HasTarget] = None

  /** Target information for annotating the underlying SRAM if it is known. */
  def underlying: Option[HasTarget] = _underlying
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
  * the inner memory with `loadMemoryFromFile` using `MemoryLoadFileType.Binary` as the file type.
  *
  * @param path The path to the binary file
  */
case class BinaryMemoryFile(path: String) extends MemoryFile(MemoryLoadFileType.Binary)

/** A hex memory file to preload an [[SRAM]] with, represented by a filesystem path. This will annotate
  * the inner memory with `loadMemoryFromFile` using `MemoryLoadFileType.Hex` as the file type.
  *
  * @param path The path to the hex file
  */
case class HexMemoryFile(path: String) extends MemoryFile(MemoryLoadFileType.Hex)

object SRAM {

  /** Generates a memory within the current module, connected to an explicit number
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

  /** Generates a memory within the current module, connected to an explicit number
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

  /** Generates a memory within the current module, connected to an explicit number
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

  /** Generates a memory within the current module, connected to an explicit number
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

  /** Generates a memory within the current module, connected to an explicit number
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

  /** Generates a memory within the current module, connected to an explicit number
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

  /** Generates a memory within the current module, connected to an explicit number
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

  /** Generates a memory within the current module, connected to an explicit number
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

    // underlying target
    val mem = autoNameRecursively("sram")(new SramTarget)

    // user-facing interface into the SRAM
    val sramIntfType = new SRAMInterface(size, tpe, numReadPorts, numWritePorts, numReadwritePorts, isVecMem)
    val _out = Wire(sramIntfType)
    _out._underlying = Some(HasTarget(mem))

    // create actual ports into firrtl memory
    val firrtlReadPorts:  Seq[FirrtlMemoryReader[_]] = sramIntfType.readPorts.map(new FirrtlMemoryReader(_))
    val firrtlWritePorts: Seq[FirrtlMemoryWriter[_]] = sramIntfType.writePorts.map(new FirrtlMemoryWriter(_))
    val firrtlReadwritePorts: Seq[FirrtlMemoryReadwriter[_]] =
      sramIntfType.readwritePorts.map(new FirrtlMemoryReadwriter(_))

    // set references to firrtl memory ports
    def nameAndSetRef(ports: Seq[Data], namePrefix: String): Seq[String] = {
      ports.zipWithIndex.map {
        case (p, idx) =>
          val name = namePrefix + idx
          p.setRef(Slot(Node(mem), name))
          name
      }
    }

    val firrtlReadPortNames = nameAndSetRef(firrtlReadPorts, "R")
    val firrtlWritePortNames = nameAndSetRef(firrtlWritePorts, "W")
    val firrtlReadwritePortNames = nameAndSetRef(firrtlReadwritePorts, "RW")

    // set bindings of firrtl memory ports
    firrtlReadPorts.foreach(_.bind(SramPortBinding(Builder.forcedUserModule, Builder.currentWhen)))
    firrtlWritePorts.foreach(_.bind(SramPortBinding(Builder.forcedUserModule, Builder.currentWhen)))
    firrtlReadwritePorts.foreach(_.bind(SramPortBinding(Builder.forcedUserModule, Builder.currentWhen)))

    // bind type so that memory type can get converted to FIRRTL
    val boundType = tpe.cloneTypeFull
    boundType.bind(FirrtlMemTypeBinding(mem))

    // create FIRRTL memory
    Builder.pushCommand(
      FirrtlMemory(
        sourceInfo,
        mem,
        boundType,
        size,
        firrtlReadPortNames,
        firrtlWritePortNames,
        firrtlReadwritePortNames
      )
    )

    // connect firrtl memory ports to user-facing interface
    for (((memReadPort, firrtlReadPort), readClock) <- _out.readPorts.zip(firrtlReadPorts).zip(readPortClocks)) {
      firrtlReadPort.addr := memReadPort.address
      firrtlReadPort.clk := readClock
      memReadPort.data := firrtlReadPort.data.asInstanceOf[Data]
      firrtlReadPort.en := memReadPort.enable
    }
    for (((memWritePort, firrtlWritePort), writeClock) <- _out.writePorts.zip(firrtlWritePorts).zip(writePortClocks)) {
      firrtlWritePort.addr := memWritePort.address
      firrtlWritePort.clk := writeClock
      firrtlWritePort.data.asInstanceOf[Data] := memWritePort.data
      firrtlWritePort.en := memWritePort.enable
      assignMask(memWritePort.data, memWritePort.mask, firrtlWritePort.getRef, "mask")
    }
    for (
      ((memReadwritePort, firrtlReadwritePort), readwriteClock) <-
        _out.readwritePorts.zip(firrtlReadwritePorts).zip(readwritePortClocks)
    ) {
      firrtlReadwritePort.addr := memReadwritePort.address
      firrtlReadwritePort.clk := readwriteClock
      firrtlReadwritePort.en := memReadwritePort.enable
      memReadwritePort.readData := firrtlReadwritePort.rdata.asInstanceOf[Data]
      firrtlReadwritePort.wdata.asInstanceOf[Data] := memReadwritePort.writeData
      firrtlReadwritePort.wmode := memReadwritePort.isWrite
      assignMask(memReadwritePort.writeData, memReadwritePort.mask, firrtlReadwritePort.getRef, "wmask")
    }

    _out
  }

  /** Assigns a given SRAM-style mask to a FIRRTL memory port mask. The FIRRTL
    * memory port mask is tied to 1 for unmasked SRAMs.
    *
    * @param memWriteDataTpe write data type to get masked
    * @param writeMaskOpt write mask to assign
    * @param firrtlMemPortRef reference to the FIRRTL memory port
    * @param maskName name of the mask in the FIRRTL memory port
    */
  private def assignMask(
    memWriteDataTpe:  Data,
    writeMaskOpt:     Option[Vec[Bool]],
    firrtlMemPortRef: Arg,
    maskName:         String
  ): Unit = {
    memWriteDataTpe match {
      case v: Vec[_] =>
        writeMaskOpt match {
          case Some(m) => assignVecMask(v, m, Slot(firrtlMemPortRef, maskName))
          case None    => assignVecMask(v, VecInit.fill(v.length)(true.B), Slot(firrtlMemPortRef, maskName))
        }
      case e => assignElementMask(e, true.B, Slot(firrtlMemPortRef, maskName))
    }

    def assignElementMask(writeData: Data, writeMask: Bool, arg: Arg)(implicit sourceInfo: SourceInfo): Unit = {
      writeData match {
        case e: Element => Builder.pushCommand(ir.Connect(sourceInfo, arg, writeMask.ref))
        case r: Record =>
          r.elements.foreach { case (name, data) => assignElementMask(data, writeMask, Slot(arg, name)) }
        case v: Vec[_] =>
          v.elementsIterator.zipWithIndex.foreach {
            case (data, idx) =>
              assignElementMask(data, writeMask, Index(arg, ILit(idx)))
          }
      }
    }

    def assignVecMask[T <: Data](
      writeData: Vec[T],
      writeMask: Vec[Bool],
      arg:       Arg
    )(
      implicit sourceInfo: SourceInfo
    ): Unit = {
      writeData.zip(writeMask).zipWithIndex.foreach {
        case ((elem, mask), idx) =>
          assignElementMask(elem, mask, Index(arg, ILit(idx)))
      }
    }
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

/** Contains fields that map from the user-facing [[MemoryReadPort]] to a
  * FIRRTL memory read port.
  *
  * @param readPort used to parameterize this class
  *
  * @note This is private because users should not directly use this bundle to
  * interact with [[SRAM]].
  */
private[chisel3] final class FirrtlMemoryReader[T <: Data](readPort: MemoryReadPort[T]) extends Bundle {
  val addr = readPort.address.cloneType
  val en = Bool()
  val clk = Clock()
  val data = Flipped(readPort.data.cloneType)
}

/** Contains fields that map from the user-facing [[MemoryWritePort]] to a
  * FIRRTL memory write port, excluding the mask, which is calculated and
  * assigned explicitly.
  *
  * @param writePort used to parameterize this class
  *
  * @note This is private because users should not directly use this bundle to
  * interact with [[SRAM]].
  */
private[chisel3] final class FirrtlMemoryWriter[T <: Data](writePort: MemoryWritePort[T]) extends Bundle {
  val addr = writePort.address.cloneType
  val en = Bool()
  val clk = Clock()
  val data = writePort.data.cloneType
}

/** Contains fields that map from the user-facing [[MemoryReadwritePort]] to a
  * FIRRTL memory read/write port, excluding the mask, which is calculated and
  * assigned explicitly.
  *
  * @param readwritePort used to parameterize this class
  *
  * @note This is private because users should not directly use this bundle to
  * interact with [[SRAM]].
  */
private[chisel3] final class FirrtlMemoryReadwriter[T <: Data](readwritePort: MemoryReadWritePort[T]) extends Bundle {
  val addr = readwritePort.address.cloneType
  val en = Bool()
  val clk = Clock()
  val rdata = Flipped(readwritePort.readData.cloneType)
  val wmode = Bool()
  val wdata = readwritePort.writeData.cloneType
}
