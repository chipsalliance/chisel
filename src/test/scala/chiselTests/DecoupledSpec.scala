// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.stage.ChiselStage
import chisel3.util.Decoupled

class DecoupledSpec extends ChiselFlatSpec {
  "Decoupled() and Decoupled.empty" should "give DecoupledIO with empty payloads" in {
    ChiselStage.elaborate(new Module {
      val io = IO(new Bundle {
        val in = Flipped(Decoupled())
        val out = Decoupled.empty
      })
      io.out <> io.in
      assert(io.asUInt.widthOption.get === 4)
    })
  }
}
