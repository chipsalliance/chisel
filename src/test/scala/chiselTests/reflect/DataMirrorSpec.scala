// SPDX-License-Identifier: Apache-2.0

package chiselTests.reflect

import chisel3._
import chisel3.reflect.DataMirror
import chiselTests.ChiselFlatSpec
import circt.stage.ChiselStage
import chisel3.util.DecoupledIO

object DataMirrorSpec {
  import org.scalatest.matchers.should.Matchers._
  class GrandChild(parent: RawModule) extends Module {
    val internal = WireInit(false.B)
    DataMirror.getParent(this) should be(Some(parent))
    DataMirror.isVisible(internal) should be(true)
  }
  class Child(parent: RawModule) extends Module {
    val inst = Module(new GrandChild(this))
    val io = IO(Input(Bool()))
    val internal = WireInit(false.B)
    DataMirror.getParent(inst) should be(Some(this))
    DataMirror.getParent(this) should be(Some(parent))
    DataMirror.isVisible(io) should be(true)
    DataMirror.isVisible(internal) should be(true)
    DataMirror.isVisible(inst.internal) should be(false)
  }
  class Parent extends Module {
    val inst = Module(new Child(this))
    inst.io := false.B
    DataMirror.getParent(inst) should be(Some(this))
    DataMirror.getParent(this) should be(None)
    DataMirror.isVisible(inst.io) should be(true)
    DataMirror.isVisible(inst.internal) should be(false)
    DataMirror.isVisible(inst.inst.internal) should be(false)
  }
}

class DataMirrorSpec extends ChiselFlatSpec {
  import DataMirrorSpec._

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
    ChiselStage.emitCHIRRTL(new MyModule)
  }

  it should "support getParent for normal modules" in {
    ChiselStage.emitCHIRRTL(new Parent)
  }

  it should "support getParent for normal modules even when used in a D/I context" in {
    import chisel3.experimental.hierarchy._
    class Top extends Module {
      val defn = Definition(new Parent)
      val inst = Instance(defn)
      DataMirror.getParent(this) should be(None)
    }
    ChiselStage.emitCHIRRTL(new Top)
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
    ChiselStage.emitCHIRRTL(new MyModule)
  }

  it should "not support name guesses for non-hardware" in {
    an[ExpectedHardwareException] should be thrownBy DataMirror.queryNameGuess(UInt(8.W))
  }

  "chiselTypeClone" should "preserve Scala type information" in {
    class MyModule extends Module {
      val in = IO(Input(UInt(8.W)))
      val out = IO(Output(DataMirror.internal.chiselTypeClone(in)))
      // The connection checks the types
      out :#= in
    }
    ChiselStage.emitCHIRRTL(new MyModule)
  }

  "isFullyAligned" should "work" in {
    class InputOutputTest extends Bundle {
      val incoming = Input(DecoupledIO(UInt(8.W)))
      val outgoing = Output(DecoupledIO(UInt(8.W)))
      val mixed = DecoupledIO(UInt(8.W))
    }
    // Top-level negative test.
    assert(!DataMirror.isFullyAligned(new InputOutputTest()))

    // Various positive tests, coerced.
    assert(DataMirror.isFullyAligned(new InputOutputTest().incoming))
    assert(DataMirror.isFullyAligned(new InputOutputTest().outgoing))
    assert(DataMirror.isFullyAligned(Input(new InputOutputTest())))
    assert(DataMirror.isFullyAligned(Input(new InputOutputTest()).outgoing))
    assert(DataMirror.isFullyAligned(Output(new InputOutputTest())))
    assert(DataMirror.isFullyAligned(Output(new InputOutputTest()).incoming))
    assert(DataMirror.isFullyAligned(Output(new InputOutputTest()).incoming.ready))

    // Negative test mixed, check positive when coerced.
    assert(!DataMirror.isFullyAligned(new InputOutputTest().mixed))
    assert(DataMirror.isFullyAligned(Input(new InputOutputTest().mixed)))

    // Check DecoupledIO directly, as well as coerced.
    assert(!DataMirror.isFullyAligned(new DecoupledIO(UInt(8.W))))
    assert(DataMirror.isFullyAligned(Input(new DecoupledIO(UInt(8.W)))))

    // Positive test, simple vector + flipped vector.
    assert(DataMirror.isFullyAligned(Vec(2, UInt(1.W))))
    assert(DataMirror.isFullyAligned(Flipped(Vec(2, UInt(1.W)))))

    // Positive test, zero-length vector of non-aligned elements.
    assert(DataMirror.isFullyAligned(Vec(0, new DecoupledIO(UInt(8.W)))))

    // Negative test: vector of flipped (?).
    assert(!DataMirror.isFullyAligned(Vec(2, Flipped(UInt(1.W)))))

    // Check empty bundle (no members).
    assert(DataMirror.isFullyAligned(new Bundle {}))
    assert(DataMirror.isFullyAligned(Flipped(new Bundle {})))

    // Check ground type.
    assert(DataMirror.isFullyAligned(UInt(8.W)))
  }

}
