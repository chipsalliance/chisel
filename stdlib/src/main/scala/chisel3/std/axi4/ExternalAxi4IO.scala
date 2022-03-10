// SPDX-License-Identifier: Apache-2.0
package chisel3.std.axi4

import chisel3._

/** This bundle follows the flat, standard Axi4 naming conventions.
  * This is useful for interacting with external IP, but should not be
  * used inside of a Chisel design since it is much less convenient to use.
  * All signals are from the perspective of the client (the source of requests).
  * Use `Flipped(new ExternalAxi4IO(...))` to obtain the signals for the server
  * (the destination of requests).
  */
class ExternalAxi4IO(params: Axi4IOParameters, userBits: Axi4UserBitsParameters = Axi4UserBitsParameters()) extends Bundle {
  // write address channel
  val AWREADY = Input(Bool())
  val AWVALID = Output(Bool())
  val AWID = Output(UInt(params.idBits.W))
  val AWADDR = Output(UInt(params.addressBits.W))

  // TODO

}
