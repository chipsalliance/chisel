// SPDX-License-Identifier: Apache-2.0

package chiselTests.experimental

import chisel3._
import chisel3.util.Valid
import chisel3.stage.ChiselStage
import chisel3.experimental.DataMirror
import chiselTests.ChiselFlatSpec

class DataMirrorSpec extends ChiselFlatSpec {
  behavior.of("DataMirror")

  def assertBinding(x: Data, io: Boolean, wire: Boolean, reg: Boolean) = {
    DataMirror.isIO(x) should be(io)
    DataMirror.isWire(x) should be(wire)
    DataMirror.isReg(x) should be(reg)
  }

  def assertIO(x: Data) = assertBinding(x, true, false, false)

  def assertWire(x: Data) = assertBinding(x, false, true, false)

  def assertReg(x: Data) = assertBinding(x, false, false, true)

  def assertNone(x: Data) = assertBinding(x, false, false, false)

  it should "validate bindings" in {
    class MyModule extends Module {
      val typ = UInt(4.W)
      val vectyp = Vec(8, UInt(4.W))
      val io = IO(new Bundle {
        val in = Input(UInt(4.W))
        val vec = Input(vectyp)
        val out = Output(UInt(4.W))
      })
      val vec = Wire(vectyp)
      val regvec = Reg(vectyp)
      val wire = Wire(UInt(4.W))
      val reg = RegNext(wire)

      assertIO(io)
      assertIO(io.in)
      assertIO(io.out)
      assertIO(io.vec(1))
      assertIO(io.vec)
      assertWire(vec)
      assertWire(vec(0))
      assertWire(wire)
      assertReg(reg)
      assertReg(regvec)
      assertReg(regvec(2))
      assertNone(typ)
      assertNone(vectyp)
    }
    ChiselStage.elaborate(new MyModule)
  }
}
