// SPDX-License-Identifier: Apache-2.0

package chisel3.util.circt

import chisel3._
import chisel3.experimental.Analog
import chisel3.util.log2Ceil

import scala.collection.immutable.SeqMap

/** Metadata from circt,
  * defined in [[https://github.com/llvm/circt/blob/ce6901d54f0898e2f3c17ade083c758e27fab6dd/lib/Dialect/FIRRTL/Transforms/CreateSiFiveMetadata.cpp#L588-L611]]
  * in the future, it should have a stable ABI definition.
  */
object CirctMemory {
  def fromObj(obj: ujson.Obj): CirctMemory =
    CirctMemory(
      obj("module_name").str,
      obj("depth").num.toInt,
      obj("width").num.toInt,
      obj("masked").bool,
      obj("read").num.toInt,
      obj("write").num.toInt,
      obj("readwrite").num.toInt,
      obj("mask_granularity").numOpt.getOrElse(0).toInt,
      obj("extra_ports").arr
        .map(_.obj)
        .map(p =>
          Port(
            p("name").str,
            p("direction").str match {
              case "in"    => Port.In
              case "out"   => Port.Out
              case "inout" => Port.Inout
            },
            p("width").num.toInt
          )
        )
        .toSeq,
      obj("hierarchy").arr.map(_.str).toSeq
    )
  def fromArr(arr: ujson.Arr): Seq[CirctMemory] =
    arr.value.map(_.obj).map(o => fromObj(o)).toSeq
  def fromFile(file: os.Path) = ujson.read(os.read(file)).arrOpt.map(a => fromArr(a))
  def interface(circtMemory: CirctMemory): CirctMemoryInterface = new CirctMemoryInterface(circtMemory)
}

/** Metadata from circt,
  * defined in [[https://github.com/llvm/circt/blob/ce6901d54f0898e2f3c17ade083c758e27fab6dd/lib/Dialect/FIRRTL/Transforms/CreateSiFiveMetadata.cpp#L588-L611]]
  * in the future, it should have a stable ABI definition.
  */
case class CirctMemory(
  moduleName:      String,
  depth:           Int,
  width:           Int,
  masked:          Boolean,
  read:            Int,
  write:           Int,
  readwrite:       Int,
  maskGranularity: Int,
  extraPorts:      Seq[Port],
  hierarchy:       Seq[String])

/** Interface for circt generated SRAM. */
class CirctMemoryInterface(parameter: CirctMemory) extends Record {
  class R extends Record {
    lazy val clock:   Clock = Input(Clock())
    lazy val address: UInt = Input(UInt(log2Ceil(parameter.depth).W))
    lazy val data:    UInt = Output(UInt(parameter.width.W))
    lazy val enable:  Bool = Input(Bool())

    def elements: SeqMap[String, Data] = SeqMap(
      "clk" -> clock,
      "addr" -> address,
      "data" -> data,
      "en" -> enable
    )
  }
  class W extends Record {
    lazy val clock:   Clock = Input(Clock())
    lazy val address: UInt = Input(UInt(log2Ceil(parameter.depth).W))
    lazy val data:    UInt = Input(UInt(parameter.width.W))
    lazy val mask:    UInt = Input(UInt(parameter.width.W))
    lazy val enable:  Bool = Input(Bool())

    def elements: SeqMap[String, Data] = SeqMap(
      "clk" -> clock,
      "addr" -> address,
      "data" -> data,
      "en" -> enable
    ) ++ Option.when(parameter.masked)("mask" -> mask)
  }
  class RW extends Record {
    lazy val clock = Input(Clock())
    lazy val address = Input(UInt(log2Ceil(parameter.depth).W))
    lazy val writeData = Input(UInt(parameter.width.W))
    lazy val writeMask = Input(UInt(parameter.width.W))
    lazy val writeEnable = Input(Bool())
    lazy val readData = Output(UInt(parameter.width.W))
    lazy val enable = Input(Bool())

    def elements: SeqMap[String, Data] = SeqMap(
      "clk" -> clock,
      "addr" -> address,
      "wdata" -> writeData,
      "wmode" -> writeEnable,
      "rdata" -> readData,
      "en" -> enable
    ) ++ Option.when(parameter.masked)("wmask" -> writeMask)
  }
  def elements: SeqMap[String, Data] =
    (Seq.tabulate(parameter.read)(i => s"R$i" -> new R) ++
      Seq.tabulate(parameter.write)(i => s"W$i" -> new R) ++
      Seq.tabulate(parameter.readwrite)(i => s"RW$i" -> new W) ++
      parameter.extraPorts.map(p => p.name -> (p.direction match {
        case Port.In => Input(UInt(p.width.W))
        case Port.Out => Output(UInt(p.width.W))
        case Port.Inout => Analog(p.width.W)
      }))
      ).to(SeqMap)
}
