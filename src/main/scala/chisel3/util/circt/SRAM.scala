// SPDX-License-Identifier: Apache-2.0

package chisel3.util.circt

import chisel3._
import chisel3.experimental.IntrinsicModule
import chisel3.experimental.{IntParam, StringParam}
import chisel3.util.log2Ceil

/** declare a standard sram port.
  * @param hasRead  does the port have a read port
  * @param hasWrite does the port have a write port
  * @param hasMask  does the port support bit level mask.
  */
case class SRAMPortInfo(hasRead: Boolean, hasWrite: Boolean, hasMask: Boolean, portDelay: Int = 1)

/** a standard SRAM port. */
class SRAMPort(
  portInfo:     SRAMPortInfo,
  addressWidth: Int,
  dataWidth:    Int)
    extends Bundle {
  val clock:      Clock = Input(Clock())
  val address:    UInt = Input(UInt(addressWidth.W))
  val chipEnable: Bool = Input(Bool())
  // TODO: use Option.when when we get rid of 2.12
  val writeEnable: Option[Bool] = if (portInfo.hasWrite) Some(Input(Bool())) else None
  val writeData:   Option[UInt] = if (portInfo.hasWrite) Some(Input(UInt(dataWidth.W))) else None
  val writeMask:   Option[UInt] = if (portInfo.hasMask && portInfo.hasWrite) Some(Input(UInt(dataWidth.W))) else None
  val readData:    Option[UInt] = if (portInfo.hasRead) Some(Output(UInt(dataWidth.W))) else None
}

/** a SRAM intrinsic to represent a SRAM.
  * @param width data width of the SRAM
  * @param depth depth of the SRAM
  * @param memoryPorts declare ports
  * @param extraIO extra IO to be added to the module(this is useful in memory BIST/DFT designs)
  * @param contents the preset contents of the memory(for FPGA, it can be used for memory initialization)
  * @param macroName optional hard macro name for PnR
  */
class SRAM(
  width:       Int,
  depth:       Int,
  memoryPorts: Seq[SRAMPortInfo],
  extraIO:     Option[Record] = None,
  contents:    Option[os.Path] = None,
  macroName:   Option[String] = None)
    extends IntrinsicModule(
      "circt_sram",
      Map(
        "width" -> IntParam(width),
        "depth" -> IntParam(depth),
        "contents" -> StringParam(contents.map(_.toString).getOrElse("")),
        "macroName" -> StringParam(macroName.getOrElse("")),
        "portDelay" -> StringParam(memoryPorts.map(_.portDelay).mkString(","))
      )
    ) {
  val ports = memoryPorts.map(info =>
    IO(new SRAMPort(info, addressWidth = log2Ceil(depth), dataWidth = width))
      .suggestName(if (info.hasRead && info.hasWrite) "readwrite" else if (info.hasRead) "read" else "write")
  )

  val extra = extraIO.map(IO(_))
  override val desiredName =
    s"SRAM${macroName.map("_" + _).getOrElse("")}w${width}x${depth}${macroName.map(n => s"_$n").getOrElse("")}"
}
