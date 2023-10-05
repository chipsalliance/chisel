// SPDX-License-Identifier: Apache-2.0

package chiselTests

import circt.stage.ChiselStage
import chisel3._
import chisel3.util._
import chisel3.testers.BasicTester
import org.scalacheck._
import scala.language.reflectiveCalls

class PassthroughModuleTester extends Module {
  val io = IO(Flipped(new PassthroughModuleIO))
  // This drives the input of a PassthroughModule
  io.in := 123.U
  assert(io.out === 123.U)
}

class VecIntegrationSpec extends ChiselPropSpec {

  property("A Reg of a Vec should operate correctly") {
    class RegTesterMod(vecSize: Int) extends Module {
      val io = IO(new Bundle {
        val in = Input(Vec(vecSize, UInt()))
        val out = Output(Vec(vecSize, UInt()))
      })
      val vecReg = RegNext(io.in, VecInit(Seq.fill(vecSize) { 0.U }))
      io.out := vecReg
    }

    class RegTester(w: Int, values: List[Int]) extends BasicTester {
      val v = VecInit(values.map(_.U(w.W)))
      val dut = Module(new RegTesterMod(values.length))
      val doneReg = RegInit(false.B)
      dut.io.in := v
      when(doneReg) {
        for ((a, b) <- dut.io.out.zip(values))
          chisel3.assert(a === b.U)
        stop()
      }.otherwise {
        doneReg := true.B
        for (a <- dut.io.out)
          chisel3.assert(a === 0.U)
      }
    }

    forAll(safeUIntN(8)) {
      case (w: Int, v: List[Int]) =>
        assertTesterPasses { new RegTester(w, v) }
    }
  }

  property("VecInit should iterate correctly") {
    class IterateTester(start: Int, len: Int)(f: UInt => UInt) extends BasicTester {
      val controlVec = VecInit(Seq.iterate(start.U, len)(f))
      val testVec = VecInit.iterate(start.U, len)(f)
      chisel3.assert(
        controlVec.asUInt === testVec.asUInt,
        cf"Expected Vec to be filled like $controlVec, instead created $testVec\n"
      )
      stop()
    }
    forAll(Gen.choose(1, 10), smallPosInts) { (start: Int, len: Int) =>
      assertTesterPasses { new IterateTester(start, len)(x => x + 50.U) }
    }
  }

  property("Regs of vecs should be usable as shift registers") {
    class ShiftRegisterTester(n: Int) extends BasicTester {
      val (cnt, wrap) = Counter(true.B, n * 2)
      val shifter = Reg(Vec(n, UInt((log2Ceil(n).max(1)).W)))
      shifter.zip(shifter.drop(1)).foreach { case (l, r) => l := r }
      shifter(n - 1) := cnt
      when(cnt >= n.asUInt) {
        val expected = cnt - n.asUInt
        chisel3.assert(shifter(0) === expected)
      }
      when(wrap) {
        stop()
      }
    }

    forAll(smallPosInts) { (n: Int) => assertTesterPasses { new ShiftRegisterTester(n) } }
  }

  property("Dynamic indexing of a Vec of Module IOs should work") {
    class ModuleIODynamicIndexTester(n: Int) extends BasicTester {
      val duts = VecInit.fill(n)(Module(new PassthroughModule).io)
      val tester = Module(new PassthroughModuleTester)

      val (cycle, done) = Counter(true.B, n)
      for ((m, i) <- duts.zipWithIndex) {
        when(cycle =/= i.U) {
          m.in := 0.U // default
          chisel3.assert(m.out === 0.U)
        }.otherwise {
          m.in := DontCare
        }
      }
      // only connect one dut per cycle
      duts(cycle) <> tester.io
      chisel3.assert(duts(cycle).out === 123.U)

      when(done) { stop() }
    }

    assertTesterPasses { new ModuleIODynamicIndexTester(4) }
  }

}
