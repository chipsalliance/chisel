// SPDX-License-Identifier: Apache-2.0

package chiselTests.reflect

import chisel3._
import chisel3.reflect.DataMirror
import chiselTests.ChiselFlatSpec
import circt.stage.ChiselStage

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

  it should "support getting name guesses even though they may change" in {
    import DataMirror.queryNameGuess
    class MyModule extends Module {
      val io = {
        val port = IO(new Bundle {
          val foo = Output(UInt(8.W))
        })
        queryNameGuess(port) should be("io_port")
        queryNameGuess(port.foo) should be("io_port.foo")
        port
      }
      queryNameGuess(io) should be("io")
      queryNameGuess(io.foo) should be("io.foo")
      io.suggestName("potato")
      queryNameGuess(io) should be("potato")
      queryNameGuess(io.foo) should be("potato.foo")
    }
    ChiselStage.elaborate(new MyModule)
  }

  it should "not support name guesses for non-hardware" in {
    an[ExpectedHardwareException] should be thrownBy DataMirror.queryNameGuess(UInt(8.W))
  }
}
