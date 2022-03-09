// SPDX-License-Identifier: Apache-2.0

package chisel3.std
import chisel3._
import chisel3.util._

case class NastiBundleParameters(
  addrBits: Int,
  dataBits: Int,
  idBits:   Int) {
  require(dataBits >= 8, s"AXI4 data bits must be >= 8 (got $dataBits)")
  require(addrBits >= 1, s"AXI4 addr bits must be >= 1 (got $addrBits)")
  require(idBits >= 1, s"AXI4 id bits must be >= 1 (got $idBits)")
  require(isPow2(dataBits), s"AXI4 data bits must be pow2 (got $dataBits)")
}

/** aka the AW/AR channel */
class NastiAddressBundle(params: NastiBundleParameters) extends Bundle {
  val id = UInt(params.idBits.W)
  val addr = UInt(params.addrBits.W)
  val len = UInt(NastiConstants.LenBits.W) // number of beats - 1
  val size = UInt(NastiConstants.SizeBits.W) // bytes in beat = 2^size
  val burst = UInt(NastiConstants.BurstBits.W)
  val lock = UInt(NastiConstants.LockBits.W)
  val cache = UInt(NastiConstants.CacheBits.W)
  val prot = UInt(NastiConstants.ProtBits.W)
  val qos = UInt(NastiConstants.QosBits.W) // 0=no QoS, bigger = higher priority
}

object NastiAddressBundle {
  def apply(params: NastiBundleParameters)(id: UInt, addr: UInt, size: UInt, len: UInt = 0.U): NastiAddressBundle = {
    val aw = Wire(new NastiAddressBundle(params))
    aw.id := id
    aw.addr := addr
    aw.len := len
    aw.size := size
    aw.burst := NastiConstants.BurstIncr
    aw.lock := false.B
    aw.cache := 0.U
    aw.prot := 0.U
    aw.qos := 0.U
    aw
  }
}

/** aka the W-channel */
class NastiWriteDataBundle(params: NastiBundleParameters) extends Bundle {
  // id removed
  val data = UInt(params.dataBits.W)
  val strb = UInt((params.dataBits / 8).W)
  val last = Bool()
}

object NastiWriteDataBundle {
  def apply(
    params: NastiBundleParameters
  )(data:   UInt,
    strb:   Option[UInt] = None,
    last:   Bool = true.B
  ): NastiWriteDataBundle = {
    val w = Wire(new NastiWriteDataBundle(params))
    w.strb := strb.getOrElse(Fill(params.dataBits / 8, 1.U))
    w.data := data
    w.last := last
    w
  }
}

/** aka the R-channel */
class NastiReadDataBundle(params: NastiBundleParameters) extends Bundle {
  val id = UInt(params.idBits.W)
  val data = UInt(params.dataBits.W)
  val resp = UInt(NastiConstants.RespBits.W)
  val last = Bool()
}

object NastiReadDataBundle {
  def apply(
    params: NastiBundleParameters
  )(id:     UInt,
    data:   UInt,
    last:   Bool = true.B,
    resp:   UInt = 0.U
  ): NastiReadDataBundle = {
    val r = Wire(new NastiReadDataBundle(params))
    r.id := id
    r.data := data
    r.last := last
    r.resp := resp
    r
  }
}

/** aka the B-channel */
class NastiWriteResponseBundle(params: NastiBundleParameters) extends Bundle {
  val id = UInt(params.idBits.W)
  val resp = UInt(NastiConstants.RespBits.W)
}

object NastiWriteResponseBundle {
  def apply(params: NastiBundleParameters)(id: UInt, resp: UInt = 0.U): NastiWriteResponseBundle = {
    val b = Wire(new NastiWriteResponseBundle(params))
    b.id := id
    b.resp := resp
    b
  }
}

class NastiBundle(params: NastiBundleParameters) extends Bundle {
  val aw = Decoupled(new NastiAddressBundle(params))
  val w = Decoupled(new NastiWriteDataBundle(params))
  val b = Flipped(Decoupled(new NastiWriteResponseBundle(params)))
  val ar = Decoupled(new NastiAddressBundle(params))
  val r = Flipped(Decoupled(new NastiReadDataBundle(params)))
}
