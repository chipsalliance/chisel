// SPDX-License-Identifier: Apache-2.0

package chiselTests.experimental

import chisel3._
import chisel3.util.Valid
import chisel3.stage.ChiselStage.emitChirrtl
import chisel3.experimental.DataMirror
import chiselTests.ChiselFlatSpec

class DataMirrorSpec extends ChiselFlatSpec {
  behavior.of("DataMirror")

  it should "validate bindings" in {
    class MyModule extends RawModule {
      val io = IO(new Bundle {
        val in = Input(UInt(8.W))
        val out = Output(UInt(8.W))
      })
      val wire = Wire(UInt(4.W))
      val reg = RegNext(wire)

      DataMirror.isIO(io) should be(true)
      DataMirror.isIO(io.in) should be(true)
      DataMirror.isIO(io.out) should be(true)
      DataMirror.isWire(wire) should be(true)
      DataMirror.isReg(reg) should be(true)

      val tpe = UInt(4.W)

      DataMirror.isIO(reg) should be(false)
      DataMirror.isWire(reg) should be(false)
      DataMirror.isReg(wire) should be(false)
      DataMirror.isReg(tpe) should be(false)
    }
  }
}
