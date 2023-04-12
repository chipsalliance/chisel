package chisel3.std.tilelink

import chisel3._
import chisel3.util.DecoupledIO

import scala.collection.immutable.SeqMap

private class NoTLCException(
  channel:       String,
  linkParameter: TileLinkLinkParameter)
    extends ChiselException(
      s"call $channel in TLBundle is not present in a TL-UL or TL-UH bus:\n $linkParameter"
    )

class TLBundle(val parameter: TileLinkLinkParameter) extends Record with experimental.AutoCloneType {
  def a: DecoupledIO[TLChannelA] =
    elements("a").asInstanceOf[DecoupledIO[TLChannelA]]

  def b: DecoupledIO[TLChannelB] =
    elements
      .getOrElse("b", throw new NoTLCException("b", parameter))
      .asInstanceOf[DecoupledIO[TLChannelB]]

  def c: DecoupledIO[TLChannelC] =
    elements
      .getOrElse("c", throw new NoTLCException("c", parameter))
      .asInstanceOf[DecoupledIO[TLChannelC]]

  def d: DecoupledIO[TLChannelD] =
    elements("d").asInstanceOf[DecoupledIO[TLChannelD]]

  def e: DecoupledIO[TLChannelE] =
    elements
      .getOrElse("e", throw new NoTLCException("e", parameter))
      .asInstanceOf[DecoupledIO[TLChannelE]]

  val elements: SeqMap[String, DecoupledIO[Bundle]] =
    SeqMap[String, DecoupledIO[Bundle]](
      "a" -> DecoupledIO(new TLChannelA(parameter.channelAParameter)),
      "d" -> Flipped(DecoupledIO(new TLChannelD(parameter.channelDParameter)))
    ) ++ (
      if (parameter.hasBCEChannels)
        Seq(
          "b" -> Flipped(DecoupledIO(new TLChannelB(parameter.channelBParameter.get))),
          "c" -> DecoupledIO(new TLChannelC(parameter.channelCParameter.get)),
          "e" -> DecoupledIO(new TLChannelE(parameter.channelEParameter.get))
        )
      else
        Seq()
    )
}

class TLChannelA(val parameter: TileLinkChannelAParameter) extends Bundle {
  private val maskWidth = parameter.dataWidth / 8
  // NOTE: this field is called a_code in TileLink spec version 1.8.1 p. 15, which is probably a typo
  val opcode:  UInt = UInt(OpCode.width)
  val param:   UInt = UInt(Param.width)
  val size:    UInt = UInt(parameter.sizeWidth.W)
  val source:  UInt = UInt(parameter.sourceWidth.W)
  val address: UInt = UInt(parameter.addressWidth.W)
  val mask:    UInt = UInt(maskWidth.W)
  val data:    UInt = UInt(parameter.dataWidth.W)
  val corrupt: Bool = Bool()
}

class TLChannelB(val parameter: TileLinkChannelBParameter) extends Bundle {
  private val maskWidth = parameter.dataWidth / 8
  val opcode:  UInt = UInt(OpCode.width)
  val param:   UInt = UInt(Param.width)
  val size:    UInt = UInt(parameter.sizeWidth.W)
  val source:  UInt = UInt(parameter.sourceWidth.W)
  val address: UInt = UInt(parameter.addressWidth.W)
  val mask:    UInt = UInt(maskWidth.W)
  val data:    UInt = UInt(parameter.dataWidth.W)
  val corrupt: Bool = Bool()
}

class TLChannelC(val parameter: TileLinkChannelCParameter) extends Bundle {
  val opcode:  UInt = UInt(OpCode.width)
  val param:   UInt = UInt(Param.width)
  val size:    UInt = UInt(parameter.sizeWidth.W)
  val source:  UInt = UInt(parameter.sourceWidth.W)
  val address: UInt = UInt(parameter.addressWidth.W)
  val data:    UInt = UInt(parameter.dataWidth.W)
  val corrupt: Bool = Bool()
}

class TLChannelD(val parameter: TileLinkChannelDParameter) extends Bundle {
  val opcode:  UInt = UInt(OpCode.width)
  val param:   UInt = UInt(Param.width)
  val size:    UInt = UInt(parameter.sizeWidth.W)
  val source:  UInt = UInt(parameter.sourceWidth.W)
  val sink:    UInt = UInt(parameter.sinkWidth.W)
  val denied:  Bool = Bool()
  val data:    UInt = UInt(parameter.dataWidth.W)
  val corrupt: Bool = Bool()
}

class TLChannelE(val parameter: TileLinkChannelEParameter) extends Bundle {
  val sink: UInt = UInt(parameter.sinkWidth.W)
}
