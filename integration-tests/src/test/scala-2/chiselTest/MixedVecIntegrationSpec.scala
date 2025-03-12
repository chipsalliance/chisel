// SPDX-License-Identifier: Apache-2.0

package chiselTests

import scala.language.reflectiveCalls

import circt.stage.ChiselStage
import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
import chisel3.util._
import org.scalatest.propspec.AnyPropSpec

class MixedVecAssignTester(w: Int, values: List[Int]) extends Module {
  val v = MixedVecInit(values.map(v => v.U(w.W)))
  for ((a, b) <- v.zip(values)) {
    assert(a === b.asUInt)
  }
  stop()
}

class MixedVecRegTester(w: Int, values: List[Int]) extends Module {
  val valuesInit = MixedVecInit(values.map(v => v.U(w.W)))
  val reg = Reg(MixedVec(chiselTypeOf(valuesInit)))

  val doneReg = RegInit(false.B)
  doneReg := true.B

  when(!doneReg) {
    // First cycle: write to reg
    reg := valuesInit
  }.otherwise {
    // Second cycle: read back from reg
    for ((a, b) <- reg.zip(values)) {
      assert(a === b.asUInt)
    }
    stop()
  }
}

class MixedVecIOPassthroughModule[T <: Data](hvec: MixedVec[T]) extends Module {
  val io = IO(new Bundle {
    val in = Input(hvec)
    val out = Output(hvec)
  })
  io.out := io.in
}

class MixedVecIOTester(boundVals: Seq[Data]) extends Module {
  val v = MixedVecInit(boundVals)
  val dut = Module(new MixedVecIOPassthroughModule(MixedVec(chiselTypeOf(v))))
  dut.io.in := v
  for ((a, b) <- dut.io.out.zip(boundVals)) {
    assert(a.asUInt === b.asUInt)
  }
  stop()
}

class MixedVecZeroEntryTester extends Module {
  def zeroEntryMixedVec: MixedVec[Data] = MixedVec(Seq.empty)

  require(zeroEntryMixedVec.getWidth == 0)

  val bundleWithZeroEntryVec = new Bundle {
    val foo = Bool()
    val bar = zeroEntryMixedVec
  }
  require(0.U.asTypeOf(bundleWithZeroEntryVec).getWidth == 1)
  require(bundleWithZeroEntryVec.getWidth == 1)

  val m = Module(new Module {
    val io = IO(Output(bundleWithZeroEntryVec))
    io.foo := false.B
  })
  WireDefault(m.io.bar)

  stop()
}

class MixedVecUIntDynamicIndexTester extends Module {
  val wire: MixedVec[UInt] = Wire(MixedVec(Seq(UInt(8.W), UInt(16.W), UInt(4.W), UInt(7.W))))
  val n = wire.length

  for (i <- 0 until n) {
    wire(i) := i.U
  }

  val vecWire = VecInit(wire: Seq[UInt])

  val (cycle, done) = Counter(true.B, n)
  assert(vecWire(cycle) === cycle)

  when(done) { stop() }
}

class MixedVecTestBundle extends Bundle {
  val x = UInt(8.W)
  val y = UInt(8.W)
}

class MixedVecSmallTestBundle extends Bundle {
  val x = UInt(3.W)
  val y = UInt(3.W)
}

class MixedVecFromVecTester extends Module {
  val wire = Wire(MixedVec(Vec(3, UInt(8.W))))
  wire := MixedVecInit(Seq(20.U, 40.U, 80.U))

  assert(wire(0) === 20.U)
  assert(wire(1) === 40.U)
  assert(wire(2) === 80.U)

  stop()
}

class MixedVecConnectWithVecTester extends Module {
  val mixedVecType = MixedVec(Vec(3, UInt(8.W)))

  val m = Module(new MixedVecIOPassthroughModule(mixedVecType))
  m.io.in := VecInit(Seq(20.U, 40.U, 80.U))
  val wire = m.io.out

  assert(wire(0) === 20.U)
  assert(wire(1) === 40.U)
  assert(wire(2) === 80.U)

  stop()
}

class MixedVecConnectWithSeqTester extends Module {
  val mixedVecType = MixedVec(Vec(3, UInt(8.W)))

  val m = Module(new MixedVecIOPassthroughModule(mixedVecType))
  m.io.in := Seq(20.U, 40.U, 80.U)
  val wire = m.io.out

  assert(wire(0) === 20.U)
  assert(wire(1) === 40.U)
  assert(wire(2) === 80.U)

  stop()
}

class MixedVecOneBitTester extends Module {
  val flag = RegInit(false.B)

  val oneBit = Reg(MixedVec(Seq(UInt(1.W))))
  when(!flag) {
    oneBit(0) := 1.U(1.W)
    flag := true.B
  }.otherwise {
    assert(oneBit(0) === 1.U)
    assert(oneBit.asUInt === 1.U)
    stop()
  }
}

class MixedVecIntegrationSpec extends AnyPropSpec with PropertyUtils with ChiselSim {
  property("MixedVec varargs API should work") {
    simulate {
      new Module {
        val wire = Wire(MixedVec(UInt(1.W), UInt(8.W)))
        wire(0) := 1.U
        wire(1) := 101.U

        chisel3.assert(wire(0) === 1.U)
        chisel3.assert(wire(1) + 1.U === 102.U)

        val wireInit = MixedVecInit(1.U, 101.U)
        chisel3.assert(wireInit(0) === 1.U)
        chisel3.assert(wireInit(1) + 1.U === 102.U)

        stop()
      }
    }(RunUntilFinished(3))
  }

  property("MixedVecs should be assignable") {
    forAll(safeUIntN(8)) { case (w: Int, v: List[Int]) =>
      simulate { new MixedVecAssignTester(w, v) }(RunUntilFinished(3))
    }
  }

  property("MixedVecs should be usable as the type for Reg()") {
    forAll(safeUIntN(8)) { case (w: Int, v: List[Int]) =>
      simulate {
        new MixedVecRegTester(w, v)
      }(RunUntilFinished(3))
    }
  }

  property("MixedVecs should be passed through IO") {
    forAll(safeUIntN(8)) { case (w: Int, v: List[Int]) =>
      simulate {
        new MixedVecIOTester(v.map(i => i.U(w.W)))
      }(RunUntilFinished(3))
    }
  }

  property("MixedVecs should work with mixed types") {
    simulate {
      new MixedVecIOTester(Seq(true.B, 168.U(8.W), 888.U(10.W), -3.S))
    }(RunUntilFinished(3))
  }

  property("MixedVecs with zero entries should compile and have zero width") {
    simulate { new MixedVecZeroEntryTester }(RunUntilFinished(3))
  }

  property("MixedVecs of UInts should be dynamically indexable (via VecInit)") {
    simulate { new MixedVecUIntDynamicIndexTester }(RunUntilFinished(5))
  }

  property("MixedVecs should be creatable from Vecs") {
    simulate { new MixedVecFromVecTester }(RunUntilFinished(3))
  }

  property("It should be possible to bulk connect a MixedVec and a Vec") {
    simulate { new MixedVecConnectWithVecTester }(RunUntilFinished(3))
  }

  property("It should be possible to bulk connect a MixedVec and a Seq") {
    simulate { new MixedVecConnectWithSeqTester }(RunUntilFinished(3))
  }

  property("MixedVecs of a single 1 bit element should compile and work") {
    simulate { new MixedVecOneBitTester }(RunUntilFinished(3))
  }
}
