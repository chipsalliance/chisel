package chisel3.std.tilelink

import chisel3.util.isPow2

/**
  * Parameter of a TileLink link bundle
  *
  * All width values are specified in bits
  *
  * @param hasBCEChannels whether the link has channel B, C and E (i.e. is a TL-C link)
  */
case class TileLinkLinkParameter(
  addressWidth:   Int,
  sourceWidth:    Int,
  sinkWidth:      Int,
  dataWidth:      Int,
  sizeWidth:      Int,
  hasBCEChannels: Boolean) {
  require(addressWidth > 0)
  require(sourceWidth > 0)
  require(sinkWidth > 0)
  require(dataWidth > 0)
  require(sizeWidth > 0)
  require(dataWidth % 8 == 0, "Width of data field must be multiples of 8")
  require(isPow2(dataWidth / 8), "Width of data field in bytes must be power of 2")

  def channelAParameter: TileLinkChannelAParameter =
    TileLinkChannelAParameter(addressWidth, sourceWidth, dataWidth, sizeWidth)
  def channelBParameter: Option[TileLinkChannelBParameter] =
    if (hasBCEChannels) Some(TileLinkChannelBParameter(addressWidth, sourceWidth, dataWidth, sizeWidth)) else None
  def channelCParameter: Option[TileLinkChannelCParameter] =
    if (hasBCEChannels) Some(TileLinkChannelCParameter(addressWidth, sourceWidth, dataWidth, sizeWidth)) else None
  def channelDParameter: TileLinkChannelDParameter =
    TileLinkChannelDParameter(sourceWidth, sinkWidth, dataWidth, sizeWidth)
  def channelEParameter: Option[TileLinkChannelEParameter] =
    if (hasBCEChannels) Some(TileLinkChannelEParameter(sinkWidth)) else None
}

case class TileLinkChannelAParameter(
  addressWidth: Int,
  sourceWidth:  Int,
  dataWidth:    Int,
  sizeWidth:    Int) {
  require(addressWidth > 0)
  require(sourceWidth > 0)
  require(dataWidth > 0)
  require(sizeWidth > 0)
  require(dataWidth % 8 == 0, "Width of data field must be multiples of 8")
  require(isPow2(dataWidth / 8), "Width of data field in bytes must be power of 2")
}

case class TileLinkChannelBParameter(
  addressWidth: Int,
  sourceWidth:  Int,
  dataWidth:    Int,
  sizeWidth:    Int) {
  require(addressWidth > 0)
  require(sourceWidth > 0)
  require(dataWidth > 0)
  require(sizeWidth > 0)
  require(dataWidth % 8 == 0, "Width of data field must be multiples of 8")
  require(isPow2(dataWidth / 8), "Width of data field in bytes must be power of 2")
}

case class TileLinkChannelCParameter(
  addressWidth: Int,
  sourceWidth:  Int,
  dataWidth:    Int,
  sizeWidth:    Int) {
  require(addressWidth > 0)
  require(sourceWidth > 0)
  require(dataWidth > 0)
  require(sizeWidth > 0)
  require(dataWidth % 8 == 0, "Width of data field must be multiples of 8")
  require(isPow2(dataWidth / 8), "Width of data field in bytes must be power of 2")
}

case class TileLinkChannelDParameter(
  sourceWidth: Int,
  sinkWidth:   Int,
  dataWidth:   Int,
  sizeWidth:   Int) {
  require(sourceWidth > 0)
  require(sinkWidth > 0)
  require(dataWidth > 0)
  require(sizeWidth > 0)
  require(dataWidth % 8 == 0, "Width of data field must be multiples of 8")
  require(isPow2(dataWidth / 8), "Width of data field in bytes must be power of 2")
}

case class TileLinkChannelEParameter(sinkWidth: Int) {
  require(sinkWidth > 0)
}
