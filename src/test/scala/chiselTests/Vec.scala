// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.core.Binding.BindingException
import chisel3.testers.BasicTester
import chisel3.util._
import org.scalacheck.Shrink

class LitTesterMod(vecSize: Int) extends Module {
  val io = IO(new Bundle {
    val out = Output(Vec(vecSize, UInt()))
  })
  io.out := Vec(vecSize, 0.U)
}

class RegTesterMod(vecSize: Int) extends Module {
  val io = IO(new Bundle {
    val in = Input(Vec(vecSize, UInt()))
    val out = Output(Vec(vecSize, UInt()))
  })
  val vecReg = Reg(init = Vec(vecSize, 0.U), next = io.in)
  io.out := vecReg
}

class IOTesterMod(vecSize: Int) extends Module {
  val io = IO(new Bundle {
    val in = Input(Vec(vecSize, UInt()))
    val out = Output(Vec(vecSize, UInt()))
  })
  io.out := io.in
}

class LitTester(w: Int, values: List[Int]) extends BasicTester {
  val dut = Module(new LitTesterMod(values.length))
  for (a <- dut.io.out)
    assert(a === 0.U)
  stop()
}

class RegTester(w: Int, values: List[Int]) extends BasicTester {
  val v = Vec(values.map(_.U(w.W)))
  val dut = Module(new RegTesterMod(values.length))
  val doneReg = RegInit(false.B)
  dut.io.in := v
  when (doneReg) {
    for ((a,b) <- dut.io.out.zip(values))
      assert(a === b.U)
    stop()
  } .otherwise {
    doneReg := true.B
    for (a <- dut.io.out)
      assert(a === 0.U)
  }
}

class IOTester(w: Int, values: List[Int]) extends BasicTester {
  val v = Vec(values.map(_.U(w.W))) // Does this need a Wire? No. It's a Vec of Lits and hence synthesizeable.
  val dut = Module(new IOTesterMod(values.length))
  dut.io.in := v
  for ((a,b) <- dut.io.out.zip(values)) {
    assert(a === b.U)
  }
  stop()
}

class IOTesterModFill(vecSize: Int) extends Module {
  // This should generate a BindingException when we attempt to wire up the Vec.fill elements
  //  since they're pure types and hence unsynthesizeable.
  val io = IO(new Bundle {
    val in = Input(Vec.fill(vecSize) {UInt()})
    val out = Output(Vec.fill(vecSize) {UInt()})
  })
  io.out := io.in
}

class ValueTester(w: Int, values: List[Int]) extends BasicTester {
  val v = Vec(values.map(_.asUInt(w.W)))
  for ((a,b) <- v.zip(values)) {
    assert(a === b.asUInt)
  }
  stop()
}

class TabulateTester(n: Int) extends BasicTester {
  val v = Vec(Range(0, n).map(i => (i*2).asUInt))
  val x = Vec(Array.tabulate(n){ i => (i*2).asUInt })
  val u = Vec.tabulate(n)(i => (i*2).asUInt)

  assert(v.asUInt() === x.asUInt())
  assert(v.asUInt() === u.asUInt())
  assert(x.asUInt() === u.asUInt())

  stop()
}

class ShiftRegisterTester(n: Int) extends BasicTester {
  val (cnt, wrap) = Counter(true.B, n*2)
  val shifter = Reg(Vec(n, UInt(log2Up(n).W)))
  (shifter, shifter drop 1).zipped.foreach(_ := _)
  shifter(n-1) := cnt
  when (cnt >= n.asUInt) {
    val expected = cnt - n.asUInt
    assert(shifter(0) === expected)
  }
  when (wrap) {
    stop()
  }
}

class HugeVecTester(n: Int) extends BasicTester {
  require(n > 0)
  val myVec = Wire(Vec(n, UInt()))
  myVec.foreach { x =>
    x := 123.U
    assert(x === 123.U)
  }
  stop()
}

class VecSpec extends ChiselPropSpec {
  // Disable shrinking on error.
  implicit val noShrinkListVal = Shrink[List[Int]](_ => Stream.empty)
  implicit val noShrinkInt = Shrink[Int](_ => Stream.empty)

  property("Vecs should be assignable") {
    forAll(safeUIntN(8)) { case(w: Int, v: List[Int]) =>
      assertTesterPasses{ new ValueTester(w, v) }
    }
  }

  property("Vecs should be passed through vec IO") {
    forAll(safeUIntN(8)) { case(w: Int, v: List[Int]) =>
      assertTesterPasses{ new IOTester(w, v) }
    }
  }

  property("Vec.fill with a pure type should generate an exception") {
    // We don't really need a sequence of random widths here, since any should throw an exception.
    forAll(safeUIntWidth) { case(w: Int) =>
      an[BindingException] should be thrownBy {
        elaborate(new IOTesterModFill(w))
      }
    }
  }

  property("A Reg of a Vec should operate correctly") {
    forAll(safeUIntN(8)) { case(w: Int, v: List[Int]) =>
      assertTesterPasses{ new RegTester(w, v) }
    }
  }

  property("A Vec of lit should operate correctly") {
    forAll(safeUIntN(8)) { case(w: Int, v: List[Int]) =>
      assertTesterPasses{ new LitTester(w, v) }
    }
  }

  property("Vecs should tabulate correctly") {
    forAll(smallPosInts) { (n: Int) => assertTesterPasses{ new TabulateTester(n) } }
  }

  property("Regs of vecs should be usable as shift registers") {
    forAll(smallPosInts) { (n: Int) => assertTesterPasses{ new ShiftRegisterTester(n) } }
  }

  property("Infering widths on huge Vecs should not cause a stack overflow") {
    assertTesterPasses { new HugeVecTester(10000) }
  }
}
