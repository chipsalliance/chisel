package chisel3.util

import chisel3._
import chisel3.internal.{Builder, NamedComponent}
import chisel3.internal.binding.{FirrtlMemTypeBinding, SramPortBinding}
import chisel3.experimental.{OpaqueType, SourceInfo}
import chisel3.experimental.hierarchy.{instantiable, public, Definition, Instance, Instantiate}
import chisel3.internal.sourceinfo.MemTransform
import chisel3.internal.firrtl.ir.{Arg, FirrtlMemory, LitIndex, Node, Ref, Slot}
import chisel3.Mem.HasVecDataType
import chisel3.util.experimental.loadMemoryFromFileInline
import chisel3.reflect.DataMirror
import firrtl.annotations.{IsMember, MemoryLoadFileType}

import scala.annotation.nowarn
import scala.language.reflectiveCalls
import chisel3.internal.firrtl.ir
import chisel3.properties.Class.ClassDefinitionOps
import chisel3.properties.{Class, ClassType, Path, Property}

import scala.collection.immutable.{ListMap, VectorMap}
import chisel3.util.experimental.{CIRCTSRAMInterface, CIRCTSRAMParameter}

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

/** Description to the SRAM, encoded by the [[chisel3.properties]] API.
  * User can access it via CIRCT API.
  */
@instantiable
final class SRAMDescription extends Class {
  val depth:           Property[BigInt] = IO(Output(Property[BigInt]()))
  val width:           Property[Int] = IO(Output(Property[Int]()))
  val masked:          Property[Boolean] = IO(Output(Property[Boolean]()))
  val read:            Property[Int] = IO(Output(Property[Int]()))
  val write:           Property[Int] = IO(Output(Property[Int]()))
  val readwrite:       Property[Int] = IO(Output(Property[Int]()))
  val maskGranularity: Property[Int] = IO(Output(Property[Int]()))
  val hierarchy:       Property[Path] = IO(Output(Property[Path]()))

  @public
  val depthIn: Property[BigInt] = IO(Input(Property[BigInt]()))
  @public
  val widthIn: Property[Int] = IO(Input(Property[Int]()))
  @public
  val maskedIn: Property[Boolean] = IO(Input(Property[Boolean]()))
  @public
  val readIn: Property[Int] = IO(Input(Property[Int]()))
  @public
  val writeIn: Property[Int] = IO(Input(Property[Int]()))
  @public
  val readwriteIn: Property[Int] = IO(Input(Property[Int]()))
  @public
  val maskGranularityIn: Property[Int] = IO(Input(Property[Int]()))
  @public
  val hierarchyIn: Property[Path] = IO(Input(Property[Path]()))

  depth := depthIn
  width := widthIn
  masked := maskedIn
  read := readIn
  write := writeIn
  readwrite := readwriteIn
  maskGranularity := maskGranularityIn
  hierarchy := hierarchyIn
}

object SRAMDescription {
  val definition: Definition[SRAMDescription] = Instantiate.definition(new SRAMDescription)
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
  * @param hasDescription Whether this interface contains an [[SRAMDescription]]
  */
class SRAMInterface[T <: Data](
  val memSize: BigInt,
  // tpe can't be directly made public as it will become a Bundle field
  tpe:                   T,
  val numReadPorts:      Int,
  val numWritePorts:     Int,
  val numReadwritePorts: Int,
  val masked:            Boolean = false,
  val hasDescription:    Boolean = false
) extends Bundle {

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

  /** Optional SRAM description to hold metadata. */
  val description: Option[Property[ClassType]] = Option.when(hasDescription)(SRAMDescription.definition.getPropertyType)
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

@nowarn("cat=deprecation")
class SRAMBlackbox(parameter: CIRCTSRAMParameter)
    extends FixedIOExtModule(new CIRCTSRAMInterface(parameter))
    with HasExtModuleInline { self =>

  private val verilogInterface: String =
    (Seq.tabulate(parameter.write)(idx =>
      Seq(
        s"// Write Port $idx",
        s"input [${log2Ceil(parameter.depth) - 1}:0] W${idx}_addr",
        s"input W${idx}_en",
        s"input W${idx}_clk",
        s"input [${parameter.width - 1}:0] W${idx}_data"
      ) ++
        Option.when(parameter.masked)(s"input [${parameter.width / parameter.maskGranularity - 1}:0] W${idx}_mask")
    ) ++
      Seq.tabulate(parameter.read)(idx =>
        Seq(
          s"// Read Port $idx",
          s"input [${log2Ceil(parameter.depth) - 1}:0] R${idx}_addr",
          s"input R${idx}_en",
          s"input R${idx}_clk",
          s"output [${parameter.width - 1}:0] R${idx}_data"
        )
      ) ++
      Seq.tabulate(parameter.readwrite)(idx =>
        Seq(
          s"// ReadWrite Port $idx",
          s"input [${log2Ceil(parameter.depth) - 1}:0] RW${idx}_addr",
          s"input RW${idx}_en",
          s"input RW${idx}_clk",
          s"input RW${idx}_wmode",
          s"input [${parameter.width - 1}:0] RW${idx}_wdata",
          s"output [${parameter.width - 1}:0] RW${idx}_rdata"
        ) ++ Option
          .when(parameter.masked)(
            s"input [${parameter.width / parameter.maskGranularity - 1}:0] RW${idx}_wmask"
          )
      )).flatten.mkString(",\n")

  private val rLogic = Seq
    .tabulate(parameter.read) { idx =>
      val prefix = s"R${idx}"
      Seq(
        s"reg _${prefix}_en;",
        s"reg [${log2Ceil(parameter.depth) - 1}:0] _${prefix}_addr;"
      ) ++
        Seq(
          s"always @(posedge ${prefix}_clk) begin // ${prefix}",
          s"_${prefix}_en <= ${prefix}_en;",
          s"_${prefix}_addr <= ${prefix}_addr;",
          s"end // ${prefix}"
        ) ++
        Some(s"assign ${prefix}_data = _${prefix}_en ? Memory[_${prefix}_addr] : ${parameter.width}'bx;")
    }
    .flatten

  private val wLogic = Seq
    .tabulate(parameter.write) { idx =>
      val prefix = s"W${idx}"
      Seq(s"always @(posedge ${prefix}_clk) begin // ${prefix}") ++
        (if (parameter.masked)
           Seq.tabulate(parameter.width / parameter.maskGranularity)(i =>
             s"if (${prefix}_en & ${prefix}_mask[${i}]) Memory[${prefix}_addr][${i * parameter.maskGranularity} +: ${parameter.maskGranularity}] <= ${prefix}_data[${(i + 1) * parameter.maskGranularity - 1}:${i * parameter.maskGranularity}];"
           )
         else
           Seq(s"if (${prefix}_en) Memory[${prefix}_addr] <= ${prefix}_data;")) ++
        Seq(s"end // ${prefix}")
    }
    .flatten

  private val rwLogic = Seq
    .tabulate(parameter.readwrite) { idx =>
      val prefix = s"RW${idx}"
      Seq(
        s"reg [${log2Ceil(parameter.depth) - 1}:0] _${prefix}_raddr;",
        s"reg _${prefix}_ren;",
        s"reg _${prefix}_rmode;"
      ) ++
        Seq(s"always @(posedge ${prefix}_clk) begin // ${prefix}") ++
        Seq(
          s"_${prefix}_raddr <= ${prefix}_addr;",
          s"_${prefix}_ren <= ${prefix}_en;",
          s"_${prefix}_rmode <= ${prefix}_wmode;"
        ) ++
        (if (parameter.masked)
           Seq.tabulate(parameter.width / parameter.maskGranularity)(i =>
             s"if(${prefix}_en & ${prefix}_wmask[${i}] & ${prefix}_wmode) Memory[${prefix}_addr][${i * parameter.maskGranularity} +: ${parameter.maskGranularity}] <= ${prefix}_wdata[${(i + 1) * parameter.maskGranularity - 1}:${i * parameter.maskGranularity}];"
           )
         else
           Seq(s"if (${prefix}_en & ${prefix}_wmode) Memory[${prefix}_addr] <= ${prefix}_wdata;")) ++
        Seq(s"end // ${prefix}") ++
        Seq(
          s"assign ${prefix}_rdata = _${prefix}_ren & ~_${prefix}_rmode ? Memory[_${prefix}_raddr] : ${parameter.width}'bx;"
        )
    }
    .flatten

  private val logic =
    (Seq(s"reg [${parameter.width - 1}:0] Memory[0:${parameter.depth - 1}];") ++ wLogic ++ rLogic ++ rwLogic)
      .mkString("\n")

  override def desiredName = parameter.moduleName

  setInline(
    desiredName + ".sv",
    s"""module ${parameter.moduleName}(
       |${verilogInterface}
       |);
       |${logic}
       |endmodule
       |""".stripMargin
  )
}

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
      1,
      1,
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
    * @param readLatency The number of cycles >= 1 between a read request and read response (applies to all ports)
    * @param writeLatency The number of cycles >= 1 between a write request and read response (applies to all ports)
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
    readLatency:       Int,
    writeLatency:      Int
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
      readLatency,
      writeLatency,
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
      1,
      1,
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
    * @param readPortClocks A sequence of clocks for each read port; and (numReadPorts + numReadwritePorts) > 0
    * @param writePortClocks A sequence of clocks for each write port; and (numWritePorts + numReadwritePorts) > 0
    * @param readwritePortClocks A sequence of clocks for each read-write port; and the above two conditions must hold
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
      1,
      1,
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
    * @param readPortClocks A sequence of clocks for each read port; and (numReadPorts + numReadwritePorts) > 0
    * @param writePortClocks A sequence of clocks for each write port; and (numWritePorts + numReadwritePorts) > 0
    * @param readwritePortClocks A sequence of clocks for each read-write port; and the above two conditions must hold
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
      1,
      1,
      Some(memoryFile),
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
    * @param readPortClocks A sequence of clocks for each read port; and (numReadPorts + numReadwritePorts) > 0
    * @param writePortClocks A sequence of clocks for each write port; and (numWritePorts + numReadwritePorts) > 0
    * @param readwritePortClocks A sequence of clocks for each read-write port; and the above two conditions must hold
    * @param readLatency The number of cycles >= 1 between a read request and read response (applies to all ports)
    * @param writeLatency The number of cycles >= 1 between a write request and read response (applies to all ports)
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
    readLatency:         Int,
    writeLatency:        Int,
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
      readLatency,
      writeLatency,
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
    implicit evidence: HasVecDataType[T],
    sourceInfo:        SourceInfo
  ): SRAMInterface[T] = {
    val clock = Builder.forcedClock
    memInterface_impl(
      size,
      tpe,
      Seq.fill(numReadPorts)(clock),
      Seq.fill(numWritePorts)(clock),
      Seq.fill(numReadwritePorts)(clock),
      1,
      1,
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
    * @param readLatency The number of cycles >= 1 between a read request and read response (applies to all ports)
    * @param writeLatency The number of cycles >= 1 between a write request and read response (applies to all ports)
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
    readLatency:       Int,
    writeLatency:      Int
  )(
    implicit evidence: HasVecDataType[T],
    sourceInfo:        SourceInfo
  ): SRAMInterface[T] = {
    val clock = Builder.forcedClock
    memInterface_impl(
      size,
      tpe,
      Seq.fill(numReadPorts)(clock),
      Seq.fill(numWritePorts)(clock),
      Seq.fill(numReadwritePorts)(clock),
      readLatency,
      writeLatency,
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
    implicit evidence: HasVecDataType[T],
    sourceInfo:        SourceInfo
  ): SRAMInterface[T] = {
    val clock = Builder.forcedClock
    memInterface_impl(
      size,
      tpe,
      Seq.fill(numReadPorts)(clock),
      Seq.fill(numWritePorts)(clock),
      Seq.fill(numReadwritePorts)(clock),
      1,
      1,
      Some(memoryFile),
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
    * @param readLatency The number of cycles >= 1 between a read request and read response (applies to all ports)
    * @param writeLatency The number of cycles >= 1 between a write request and read response (applies to all ports)
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
    readLatency:       Int,
    writeLatency:      Int,
    memoryFile:        MemoryFile
  )(
    implicit evidence: HasVecDataType[T],
    sourceInfo:        SourceInfo
  ): SRAMInterface[T] = {
    val clock = Builder.forcedClock
    memInterface_impl(
      size,
      tpe,
      Seq.fill(numReadPorts)(clock),
      Seq.fill(numWritePorts)(clock),
      Seq.fill(numReadwritePorts)(clock),
      readLatency,
      writeLatency,
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
    implicit evidence: HasVecDataType[T],
    sourceInfo:        SourceInfo
  ): SRAMInterface[T] =
    memInterface_impl(
      size,
      tpe,
      readPortClocks,
      writePortClocks,
      readwritePortClocks,
      1,
      1,
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
    * @param readLatency The number of cycles >= 1 between a read request and read response (applies to all ports)
    * @param writeLatency The number of cycles >= 1 between a write request and read response (applies to all ports)
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
    readLatency:         Int,
    writeLatency:        Int
  )(
    implicit evidence: HasVecDataType[T],
    sourceInfo:        SourceInfo
  ): SRAMInterface[T] =
    memInterface_impl(
      size,
      tpe,
      readPortClocks,
      writePortClocks,
      readwritePortClocks,
      readLatency,
      writeLatency,
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
    implicit evidence: HasVecDataType[T],
    sourceInfo:        SourceInfo
  ): SRAMInterface[T] =
    memInterface_impl(
      size,
      tpe,
      readPortClocks,
      writePortClocks,
      readwritePortClocks,
      1,
      1,
      Some(memoryFile),
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
    * @param readLatency The number of cycles >= 1 between a read request and read response (applies to all ports)
    * @param writeLatency The number of cycles >= 1 between a write request and read response (applies to all ports)
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
    readLatency:         Int,
    writeLatency:        Int,
    memoryFile:          MemoryFile
  )(
    implicit evidence: HasVecDataType[T],
    sourceInfo:        SourceInfo
  ): SRAMInterface[T] =
    memInterface_impl(
      size,
      tpe,
      readPortClocks,
      writePortClocks,
      readwritePortClocks,
      readLatency,
      writeLatency,
      Some(memoryFile),
      Some(evidence),
      sourceInfo
    )

  private def memInterface_blackbox_impl[T <: Data](
    size:                BigInt,
    tpe:                 T,
    readPortClocks:      Seq[Clock],
    writePortClocks:     Seq[Clock],
    readwritePortClocks: Seq[Clock],
    memoryFile:          Option[MemoryFile],
    evidenceOpt:         Option[HasVecDataType[T]],
    sourceInfo:          SourceInfo
  ): SRAMInterface[T] = {
    val numReadPorts = readPortClocks.size
    val numWritePorts = writePortClocks.size
    val numReadwritePorts = readwritePortClocks.size
    val enableMask = evidenceOpt.isDefined
    val isValidSRAM = ((numReadPorts + numReadwritePorts) > 0) && ((numWritePorts + numReadwritePorts) > 0)
    val maskGranularity = tpe match {
      case vec: Vec[_] if enableMask => vec.sample_element.getWidth
      case _ => 0
    }

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

    val mem = Instantiate(
      new SRAMBlackbox(
        new CIRCTSRAMParameter(
          s"sram_${numReadPorts}R_${numWritePorts}W_${numReadwritePorts}RW_${maskGranularity}M_${size}x${tpe.getWidth}",
          numReadPorts,
          numWritePorts,
          numReadwritePorts,
          size.intValue,
          tpe.getWidth,
          maskGranularity
        )
      )
    )

    implicit class SRAMInstanceMethods(underlying: Instance[SRAMBlackbox]) {
      implicit val mg: internal.MacroGenerated = new chisel3.internal.MacroGenerated {}
      def io = underlying._lookup(_.io)
    }

    val sramReadPorts = Seq.tabulate(numReadPorts)(i => mem.io.R(i))
    val sramWritePorts = Seq.tabulate(numWritePorts)(i => mem.io.W(i))
    val sramReadwritePorts = Seq.tabulate(numReadwritePorts)(i => mem.io.RW(i))

    val includeMetadata = Builder.includeUtilMetadata

    val out = Wire(
      new SRAMInterface(size, tpe, numReadPorts, numWritePorts, numReadwritePorts, enableMask, includeMetadata)
    )

    out.readPorts.zip(sramReadPorts).zip(readPortClocks).map { case ((intfReadPort, sramReadPort), readClock) =>
      sramReadPort.address := intfReadPort.address
      sramReadPort.clock := readClock
      intfReadPort.data := sramReadPort.data.asTypeOf(tpe)
      sramReadPort.enable := intfReadPort.enable
    }
    out.writePorts.zip(sramWritePorts).zip(writePortClocks).map { case ((intfWritePort, sramWritePort), writeClock) =>
      sramWritePort.address := intfWritePort.address
      sramWritePort.clock := writeClock
      sramWritePort.data := intfWritePort.data.asUInt
      sramWritePort.enable := intfWritePort.enable
      sramWritePort.mask match {
        case Some(mask) => mask := intfWritePort.mask.get.asUInt
        case None       => assert(intfWritePort.mask.isEmpty)
      }
    }
    out.readwritePorts.zip(sramReadwritePorts).zip(readwritePortClocks).map {
      case ((intfReadwritePort, sramReadwritePort), readwriteClock) =>
        sramReadwritePort.address := intfReadwritePort.address
        sramReadwritePort.clock := readwriteClock
        sramReadwritePort.enable := intfReadwritePort.enable
        intfReadwritePort.readData := sramReadwritePort.readData.asTypeOf(tpe)
        sramReadwritePort.writeData := intfReadwritePort.writeData.asUInt
        sramReadwritePort.writeEnable := intfReadwritePort.isWrite
        sramReadwritePort.writeMask match {
          case Some(mask) => mask := intfReadwritePort.mask.get.asUInt
          case None       => assert(intfReadwritePort.mask.isEmpty)
        }
    }

    out.description.foreach { description =>
      val descriptionInstance: Instance[SRAMDescription] = Instantiate(new SRAMDescription)
      descriptionInstance.depthIn := Property(size)
      descriptionInstance.widthIn := Property(tpe.getWidth)
      descriptionInstance.maskedIn := Property(enableMask)
      descriptionInstance.readIn := Property(numReadPorts)
      descriptionInstance.writeIn := Property(numWritePorts)
      descriptionInstance.readwriteIn := Property(numReadwritePorts)
      descriptionInstance.maskGranularityIn := Property(maskGranularity)
      descriptionInstance.hierarchyIn := Property(Path(mem.toTarget))
      description := descriptionInstance.getPropertyReference
    }
    out
  }

  private def memInterface_impl[T <: Data](
    size:                BigInt,
    tpe:                 T,
    readPortClocks:      Seq[Clock],
    writePortClocks:     Seq[Clock],
    readwritePortClocks: Seq[Clock],
    readLatency:         Int,
    writeLatency:        Int,
    memoryFile:          Option[MemoryFile],
    evidenceOpt:         Option[HasVecDataType[T]],
    sourceInfo:          SourceInfo
  ): SRAMInterface[T] = {
    // Validate latency parameters
    require(readLatency >= 1, s"readLatency must be >= 1, got $readLatency")
    require(writeLatency >= 1, s"writeLatency must be >= 1, got $writeLatency")

    if (Builder.useSRAMBlackbox)
      return memInterface_blackbox_impl(
        size,
        tpe,
        readPortClocks,
        writePortClocks,
        readwritePortClocks,
        memoryFile,
        evidenceOpt,
        sourceInfo
      )

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
    val mem = withName("sram")(new SramTarget)

    val includeMetadata = Builder.includeUtilMetadata

    // user-facing interface into the SRAM
    val sramIntfType =
      new SRAMInterface(size, tpe, numReadPorts, numWritePorts, numReadwritePorts, isVecMem, includeMetadata)
    val _out = Wire(sramIntfType)
    _out._underlying = Some(HasTarget(mem))

    // create actual ports into firrtl memory
    val firrtlReadPorts:  Seq[FirrtlMemoryReader[_]] = sramIntfType.readPorts.map(new FirrtlMemoryReader(_))
    val firrtlWritePorts: Seq[FirrtlMemoryWriter[_]] = sramIntfType.writePorts.map(new FirrtlMemoryWriter(_))
    val firrtlReadwritePorts: Seq[FirrtlMemoryReadwriter[_]] =
      sramIntfType.readwritePorts.map(new FirrtlMemoryReadwriter(_))

    // set references to firrtl memory ports
    def nameAndSetRef(ports: Seq[Data], namePrefix: String): Seq[String] = {
      ports.zipWithIndex.map { case (p, idx) =>
        val name = namePrefix + idx
        p.setRef(Slot(Node(mem), name))
        name
      }
    }

    val firrtlReadPortNames = nameAndSetRef(firrtlReadPorts, "R")
    val firrtlWritePortNames = nameAndSetRef(firrtlWritePorts, "W")
    val firrtlReadwritePortNames = nameAndSetRef(firrtlReadwritePorts, "RW")

    // set bindings of firrtl memory ports
    firrtlReadPorts.foreach(_.bind(SramPortBinding(Builder.forcedUserModule, Builder.currentBlock)))
    firrtlWritePorts.foreach(_.bind(SramPortBinding(Builder.forcedUserModule, Builder.currentBlock)))
    firrtlReadwritePorts.foreach(_.bind(SramPortBinding(Builder.forcedUserModule, Builder.currentBlock)))

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
        firrtlReadwritePortNames,
        readLatency,
        writeLatency
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
      assignMask(firrtlWritePort.mask, memWritePort.mask)
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
      assignMask(firrtlReadwritePort.wmask, memReadwritePort.mask)
    }

    _out.description.foreach { description =>
      val descriptionInstance: Instance[SRAMDescription] = Instantiate(new SRAMDescription)
      descriptionInstance.depthIn := Property(size)
      descriptionInstance.widthIn := Property(tpe.getWidth)
      descriptionInstance.maskedIn := Property(isVecMem)
      descriptionInstance.readIn := Property(numReadPorts)
      descriptionInstance.writeIn := Property(numWritePorts)
      descriptionInstance.readwriteIn := Property(numReadwritePorts)
      descriptionInstance.maskGranularityIn := Property(
        Option
          .when(isVecMem)(tpe match {
            case t: Vec[_] => t.sample_element.getWidth
          })
          .getOrElse(0)
      )
      descriptionInstance.hierarchyIn := Property(Path(mem))
      description := descriptionInstance.getPropertyReference
    }
    ModulePrefixAnnotation.annotate(mem)
    _out
  }

  // There appears to be a bug in ScalaDoc where you cannot use macro-generated methods in the same
  // compilation unit as the macro-generated type. This means that any use of Definition[_] or
  // Instance[_] of the above classes within this compilation unit breaks ScalaDoc generation. This
  // issue appears to be similar to https://stackoverflow.com/questions/42684101 but applying the
  // specific mitigation did not seem to work.  As a workaround, we simply write the extension methods
  // that are generated by the @instantiable macro so that we can use them here.
  implicit class SRAMDescriptionInstanceMethods(underlying: Instance[SRAMDescription]) {
    implicit val mg:       internal.MacroGenerated = new chisel3.internal.MacroGenerated {}
    def depthIn:           Property[BigInt] = underlying._lookup(_.depthIn)
    def widthIn:           Property[Int] = underlying._lookup(_.widthIn)
    def maskedIn:          Property[Boolean] = underlying._lookup(_.maskedIn)
    def readIn:            Property[Int] = underlying._lookup(_.readIn)
    def writeIn:           Property[Int] = underlying._lookup(_.writeIn)
    def readwriteIn:       Property[Int] = underlying._lookup(_.readwriteIn)
    def maskGranularityIn: Property[Int] = underlying._lookup(_.maskGranularityIn)
    def hierarchyIn:       Property[Path] = underlying._lookup(_.hierarchyIn)
  }

  /** Assigns a given SRAM-style mask to a FIRRTL memory port mask. The FIRRTL
    * memory port mask is tied to 1 for unmasked SRAMs.
    *
    * @param memWriteDataTpe write data type to get masked
    * @param writeMaskOpt write mask to assign
    */
  private def assignMask(
    writeMask:  SramMask,
    maskSource: Option[Vec[Bool]]
  ): Unit = {
    maskSource match {
      case None =>
        // Write all 1s, all leaves are Bools
        for (mask <- DataMirror.collectMembers(writeMask) { case b: Bool => b }) {
          mask := true.B
        }
      case Some(source) =>
        for ((maskElt, value) <- writeMask.vecElements.zip(source)) {
          // All leaves are Bools, write the value from the maskOpt
          for (leaf <- DataMirror.collectMembers(maskElt) { case b: Bool => b }) {
            leaf := value
          }
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

/** Type representing the mask of an SRAM
  *
  * This type shadows the data type. It matches the structure of Aggregates with all leaves replaced with Bools.
  */
private[chisel3] class SramMask(gen: Data) extends Record with OpaqueType {

  override def opaqueType = gen match {
    case r: Record => r._isOpaqueType
    case _ => true
  }

  val elements = gen match {
    case e: Element => ListMap("" -> Bool())
    case v: Vec[_]  => ListMap("" -> Vec(v.length, new SramMask(v.sample_element)))
    case r: Record  => r.elements.map { case (name, tpe) => name -> new SramMask(tpe) }
  }

  /** Used to assert that the SramMask and process its elements */
  def vecElements: Vec[SramMask] = gen match {
    case v: Vec[_] => elements.head._2.asInstanceOf[Vec[SramMask]]
    case _ => Builder.exception(s"Internal Error! SramMask.vecElements called on non-Vec type $gen!")
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
  val mask = new SramMask(data)
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
  val wmask = new SramMask(wdata)
}
