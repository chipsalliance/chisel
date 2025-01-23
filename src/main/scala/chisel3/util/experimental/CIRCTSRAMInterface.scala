// SPDX-License-Identifier: Apache-2.0

package chisel3.util.experimental

import scala.collection.immutable.SeqMap

import chisel3._
import chisel3.experimental.BaseModule
import chisel3.experimental.hierarchy.Instance

import chisel3.util.log2Ceil

object CIRCTSRAMParameter {
  implicit val rw: upickle.default.ReadWriter[CIRCTSRAMParameter] = upickle.default.macroRW
}

case class CIRCTSRAMParameter(
  moduleName:      String,
  read:            Int,
  write:           Int,
  readwrite:       Int,
  depth:           Int,
  width:           Int,
  maskGranularity: Int
) {
  def masked: Boolean = maskGranularity != 0
}

class CIRCTSRAMReadPort(memoryParameter: CIRCTSRAMParameter) extends Record {
  val clock = Input(Clock())
  val address = Input(UInt(log2Ceil(memoryParameter.depth).W))
  val data = Output(UInt(memoryParameter.width.W))
  val enable = Input(Bool())

  // Records store elements in reverse order
  val elements: SeqMap[String, Data] = SeqMap(
    "addr" -> address,
    "en" -> enable,
    "clk" -> clock,
    "data" -> data
  ).toSeq.reverse.to(SeqMap)
}

class CIRCTSRAMReadWritePort(memoryParameter: CIRCTSRAMParameter) extends Record {
  val clock = Input(Clock())
  val address = Input(UInt(log2Ceil(memoryParameter.depth).W))
  val writeData = Input(UInt(memoryParameter.width.W))
  val writeMask =
    Option.when(memoryParameter.masked)(Input(UInt((memoryParameter.width / memoryParameter.maskGranularity).W)))
  val writeEnable = Input(Bool())
  val readData = Output(UInt(memoryParameter.width.W))
  val enable = Input(Bool())

  // Records store elements in reverse order
  val elements: SeqMap[String, Data] = (SeqMap(
    "addr" -> address,
    "en" -> enable,
    "clk" -> clock,
    "wmode" -> writeEnable,
    "wdata" -> writeData,
    "rdata" -> readData
  ) ++ Option.when(memoryParameter.masked)("wmask" -> writeMask.get)).toSeq.reverse.to(SeqMap)
}

class CIRCTSRAMWritePort(memoryParameter: CIRCTSRAMParameter) extends Record {
  val clock = Input(Clock())
  val address = Input(UInt(log2Ceil(memoryParameter.depth).W))
  val data = Input(UInt(memoryParameter.width.W))
  val mask =
    Option.when(memoryParameter.masked)(Input(UInt((memoryParameter.width / memoryParameter.maskGranularity).W)))
  val enable = Input(Bool())

  // Records store elements in reverse order
  val elements: SeqMap[String, Data] = (SeqMap(
    "addr" -> address,
    "en" -> enable,
    "clk" -> clock,
    "data" -> data
  ) ++ Option.when(memoryParameter.masked)("mask" -> mask.get)).toSeq.reverse.to(SeqMap)
}

class CIRCTSRAMInterface(memoryParameter: CIRCTSRAMParameter) extends Record {
  def R(idx: Int) =
    elements.getOrElse(s"R$idx", throw new Exception(s"Cannot get port R$idx")).asInstanceOf[CIRCTSRAMReadPort]
  def RW(idx: Int) =
    elements.getOrElse(s"RW$idx", throw new Exception(s"Cannot get port RW$idx")).asInstanceOf[CIRCTSRAMReadWritePort]
  def W(idx: Int) =
    elements.getOrElse(s"W$idx", throw new Exception(s"Cannot get port W$idx")).asInstanceOf[CIRCTSRAMWritePort]

  // Records store elements in reverse order
  val elements: SeqMap[String, Data] =
    (Seq.tabulate(memoryParameter.read)(i => s"R$i" -> new CIRCTSRAMReadPort(memoryParameter)) ++
      Seq.tabulate(memoryParameter.readwrite)(i => s"RW$i" -> new CIRCTSRAMReadWritePort(memoryParameter)) ++
      Seq.tabulate(memoryParameter.write)(i => s"W$i" -> new CIRCTSRAMWritePort(memoryParameter))).reverse
      .to(SeqMap)
}

abstract class CIRCTSRAM[T <: RawModule](memoryParameter: CIRCTSRAMParameter)
    extends FixedIORawModule[CIRCTSRAMInterface](new CIRCTSRAMInterface(memoryParameter)) {
  override def desiredName: String = memoryParameter.moduleName
  val memoryInstance: Instance[_ <: BaseModule]
}
