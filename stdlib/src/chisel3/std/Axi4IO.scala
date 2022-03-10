// SPDX-License-Identifier: Apache-2.0

package chisel3.std
import chisel3._
import chisel3.experimental.ChiselEnum
import chisel3.util._

case class Axi4IOParameters(
  addressBits: Int,
  dataBits: Int,
  idBits:   Int,
  enableRegionBits: Boolean = false) {
  require(dataBits >= 8, s"AXI4 data bits must be >= 8 (got $dataBits)")
  require(addressBits >= 1, s"AXI4 addr bits must be >= 1 (got $addressBits)")
  require(idBits >= 1, s"AXI4 id bits must be >= 1 (got $idBits)")
  require(isPow2(dataBits), s"AXI4 data bits must be pow2 (got $dataBits)")
}

case class Axi4UserBitsParameters(
  writeAddress: Int = 0,
  writeData: Int = 0,
  writeResponse: Int = 0,
  readAddress: Int = 0,
  readData: Int = 0,
) {
  require(writeAddress >= 0, "number of write address user bits must be non-negative")
  require(writeData >= 0, "number of write data user bits must be non-negative")
  require(writeResponse >= 0, "number of write response user bits must be non-negative")
  require(readAddress >= 0, "number of read address user bits must be non-negative")
  require(readData >= 0, "number of read data user bits must be non-negative")
}


object Axi4BurstType extends ChiselEnum {
  val Fixed     = Value("b00".U(2.W))
  val Increment = Value("b01".U(2.W))
  val Warp      = Value("b10".U(2.W))
}

object Axi4LockType extends ChiselEnum {
  val NormalAccess    = Value(0.U(1.W))
  val ExclusiveAccess = Value(0.U(1.W))
}

class Axi4MemoryType extends Bundle {
  val writeAlloc = Bool() // bit 3
  val readAlloc  = Bool() // bit 2
  val modifiable = Bool() // bit 1
  val bufferable = Bool() // bit 0
}

class Axi4AccessPermission extends Bundle {
  /** fetch=1: instruction access, fetch=0: data access */
  val fetch      = Bool() // bit 2
  val nonSecure  = Bool() // bit 1
  val privileged = Bool() // bit 0
}

object Axi4ResponseType extends ChiselEnum {
  val Okay          = Value("b00".U(2.W))
  val ExclusiveOkay = Value("b01".U(2.W))
  val ServerError   = Value("b10".U(2.W))
  val DecodeError   = Value("b11".U(2.W))
}

class Axi4AddressChannel(params: Axi4IOParameters, userBits: Int = 0) extends Bundle {
  require(userBits >= 0, "number of user bits must be non-negative")
  val id = UInt(params.idBits.W)
  val address = UInt(params.addressBits.W)
  /** Burst length. 1 to 16 for Axi3. 1 to 256 for Axi4 Incr burst. */
  val length = UInt(8.W)
  /** Size of the transfer. 2**size in bytes. */
  val size = UInt(3.W)
  val burst = Axi4BurstType()
  val lock = Axi4LockType()
  val cache = new Axi4MemoryType
  val prot = new Axi4AccessPermission
  val qos = UInt(4.W)
  private val regionWidth = if(params.enableRegionBits) { 4 } else { 0 }
  val region = UInt(regionWidth.W)
  val user = UInt(userBits.W)
}

class Axi4WriteDataChannel(params: Axi4IOParameters, userBits: Int = 0) extends Bundle {
  val id = UInt(params.idBits.W)
  val data = UInt(params.dataBits.W)
  val strobes = UInt((params.dataBits / 8).W)
  val last = Bool()
  val user = UInt(userBits.W)
}

class Axi4WriteResponseChannel(params: Axi4IOParameters, userBits: Int = 0) extends Bundle {
  val id = UInt(params.idBits.W)
  val resp = Axi4ResponseType()
  val user = UInt(userBits.W)
}

class Axi4ReadDataChannel(params: Axi4IOParameters, userBits: Int = 0) extends Bundle {
  val id = UInt(params.idBits.W)
  val data = UInt(params.dataBits.W)
  val resp = Axi4ResponseType()
  val last = Bool()
  val user = UInt(userBits.W)
}


/** Axi4 signals from the perspective of the client (the source of requests).
  * Use `Flipped(new Axi4IO(...))` to obtain the signals for the server
  * (the destination of requests).
  * */
class Axi4IO(params: Axi4IOParameters, userBits: Axi4UserBitsParameters = Axi4UserBitsParameters()) extends Bundle {
  val writeAddress = Irrevocable(new Axi4AddressChannel(params, userBits.writeAddress))
  val writeData = Irrevocable(new Axi4WriteDataChannel(params, userBits.writeData))
  val writeResponse = Flipped(Irrevocable(new Axi4WriteResponseChannel(params, userBits.writeResponse)))
  val readAddress = Irrevocable(new Axi4AddressChannel(params, userBits.readAddress))
  val readData = Flipped(Irrevocable(new Axi4ReadDataChannel(params, userBits.readData)))
}
