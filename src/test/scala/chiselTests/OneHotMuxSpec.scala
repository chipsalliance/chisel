// See LICENSE for license details.

package chiselTests

import Chisel.testers.BasicTester
import chisel3._
import chisel3.experimental.FixedPoint
import chisel3.util.Mux1H
import org.scalatest._

//scalastyle:off magic.number

class OneHotMuxSpec extends FreeSpec with Matchers with ChiselRunners {
  "simple one hot mux with uint should work" in {
    assertTesterPasses(new SimpleOneHotTester)
  }
  "simple one hot mux with sint should work" in {
    assertTesterPasses(new SIntOneHotTester)
  }
  "simple one hot mux with fixed point should work" in {
    assertTesterPasses(new FixedPointOneHotTester)
  }
  "simple one hot mux with all same fixed point should work" in {
    assertTesterPasses(new AllSameFixedPointOneHotTester)
  }
  "simple one hot mux with all same parameterized sint values should work" in {
    val values: Seq[SInt] = Seq((-3).S, (-5).S, (-7).S, (-11).S)
    assertTesterPasses(new ParameterizedOneHotTester(values, SInt(8.W), -5.S(8.W)))
  }
  "simple one hot mux with all same parameterized aggregates containing fixed values should work" in {
    assertTesterPasses(new ParameterizedAggregateOneHotTester)
  }
  "simple one hot mux with all aggregates containing inferred width fixed values should NOT work" in {
    intercept[ChiselException] {
      assertTesterPasses(new InferredWidthAggregateOneHotTester)
    }
  }
  "simple one hot mux with all fixed width bundles but with different bundles should Not work" in {
    intercept[IllegalArgumentException] {
      assertTesterPasses(new DifferentBundleOneHotTester)
    }
  }
}

class SimpleOneHotTester extends BasicTester {
  val out = Wire(UInt())
  out := Mux1H(Seq(
    false.B -> 2.U,
    false.B -> 4.U,
    true.B  -> 8.U,
    false.B -> 11.U
  ))

  assert(out === 8.U)

  stop()
}

class SIntOneHotTester extends BasicTester {
  val out = Wire(SInt())
  out := Mux1H(Seq(
    false.B -> (-3).S,
    true.B  -> (-5).S,
    false.B -> (-7).S,
    false.B -> (-11).S
  ))

  assert(out === (-5).S)

  stop()
}

class FixedPointOneHotTester extends BasicTester {
  val out = Wire(FixedPoint(8.W, 4.BP))

  out := Mux1H(Seq(
    false.B -> (-1.5).F(1.BP),
    true.B  -> (-2.25).F(2.BP),
    false.B -> (-4.125).F(3.BP),
    false.B -> (-11.625).F(3.BP)
  ))

  assert(out === (-2.25).F(4.BP))

  stop()
}

class AllSameFixedPointOneHotTester extends BasicTester {
  val out = Wire(FixedPoint(12.W, 3.BP))

  out := Mux1H(Seq(
    false.B -> (-1.5).F(12.W, 3.BP),
    true.B  -> (-2.25).F(12.W, 3.BP),
    false.B -> (-4.125).F(12.W, 3.BP),
    false.B -> (-11.625).F(12.W, 3.BP)
  ))

  assert(out === (-2.25).F(14.W, 4.BP))

  stop()
}

class ParameterizedOneHotTester[T <: Data](values: Seq[T], outGen: T, expected: T) extends BasicTester {
  val dut = Module(new ParameterizedOneHot(values, outGen))
  dut.io.selectors(0) := false.B
  dut.io.selectors(1) := true.B
  dut.io.selectors(2) := false.B
  dut.io.selectors(3) := false.B

  assert(dut.io.out.asUInt() === expected.asUInt())

  stop()
}

class Agg1 extends Bundle {
  val v = Vec(2, FixedPoint(8.W, 4.BP))
  val a = new Bundle {
    val f1 = FixedPoint(7.W, 3.BP)
    val f2 = FixedPoint(9.W, 5.BP)
  }
}

object Agg1 extends HasMakeLit[Agg1] {
  def makeLit(n: Int): Agg1 = {
    val x = n.toDouble / 4.0
    val (d: Double, e: Double, f: Double, g: Double) = (x, x * 2.0, x * 3.0, x * 4.0)

    val w = Wire(new Agg1)
    w.v(0) := d.F(4.BP)
    w.v(1) := e.F(4.BP)
    w.a.f1 := f.F(3.BP)
    w.a.f2 := g.F(5.BP)
    w
  }
}
class Agg2 extends Bundle {
  val v = Vec(2, FixedPoint(8.W, 4.BP))
  val a = new Bundle {
    val f1 = FixedPoint(7.W, 3.BP)
    val f2 = FixedPoint(9.W, 5.BP)
  }
}

object Agg2 extends HasMakeLit[Agg2] {
  def makeLit(n: Int): Agg2 = {
    val x = n.toDouble / 4.0
    val (d: Double, e: Double, f: Double, g: Double) = (x, x * 2.0, x * 3.0, x * 4.0)

    val w = Wire(new Agg2)
    w.v(0) := d.F(4.BP)
    w.v(1) := e.F(4.BP)
    w.a.f1 := f.F(3.BP)
    w.a.f2 := g.F(5.BP)
    w
  }
}

class ParameterizedAggregateOneHotTester extends BasicTester {
  val values = (0 until 4).map { n => Agg1.makeLit(n) }

  val dut = Module(new ParameterizedAggregateOneHot(Agg1, new Agg1))
  dut.io.selectors(0) := false.B
  dut.io.selectors(1) := true.B
  dut.io.selectors(2) := false.B
  dut.io.selectors(3) := false.B

  assert(dut.io.out.asUInt() === values(1).asUInt())

  stop()
}

trait HasMakeLit[T] {
  def makeLit(n: Int): T
}

class ParameterizedOneHot[T <: Data](values: Seq[T], outGen: T) extends Module {
  val io = IO(new Bundle {
    val selectors = Input(Vec(4, Bool()))
    val out = Output(outGen)
  })

  val terms = io.selectors.zip(values)
  io.out := Mux1H(terms)
}

class ParameterizedAggregateOneHot[T <: Data](valGen: HasMakeLit[T], outGen: T) extends Module {
  val io = IO(new Bundle {
    val selectors = Input(Vec(4, Bool()))
    val out = Output(outGen)
  })


  val values = (0 until 4).map { n => valGen.makeLit(n) }
  val terms = io.selectors.zip(values)
  io.out := Mux1H(terms)
}

class Bundle1 extends Bundle {
  val a = FixedPoint()
  val b = new Bundle {
    val c = FixedPoint()
  }
}

class InferredWidthAggregateOneHotTester extends BasicTester {
  val b0 = Wire(new Bundle1)
  b0.a := -0.25.F(2.BP)
  b0.b.c := -0.125.F(3.BP)

  val b1 = Wire(new Bundle1)
  b1.a := -0.0625.F(3.BP)
  b1.b.c := -0.03125.F(4.BP)

  val b2 = Wire(new Bundle1)
  b2.a := -0.015625.F(5.BP)
  b2.b.c := -0.0078125.F(6.BP)

  val b3 = Wire(new Bundle1)
  b3.a := -0.0078125.F(7.BP)
  b3.b.c := -0.00390625.F(8.BP)

  val o1 = Mux1H(Seq(
    false.B -> b0,
    false.B -> b1,
    true.B -> b2,
    false.B -> b3
  ))

  assert(o1.a === -0.015625.F(5.BP))
  assert(o1.b.c === -0.0078125.F(6.BP))

  val o2 = Mux1H(Seq(
    false.B -> b0,
    true.B -> b1,
    false.B -> b2,
    false.B -> b3
  ))

  assert(o2.a === -0.0625.F(3.BP))
  assert(o2.b.c === -0.03125.F(4.BP))

  stop()
}

class Bundle2 extends Bundle {
  val a = FixedPoint(10.W, 4.BP)
  val b = new Bundle {
    val c = FixedPoint(10.W, 4.BP)
  }
}

class Bundle3 extends Bundle {
  val a = FixedPoint(10.W, 4.BP)
  val b = new Bundle {
    val c = FixedPoint(10.W, 4.BP)
  }
}

class DifferentBundleOneHotTester extends BasicTester {
  val b0 = Wire(new Bundle2)
  b0.a := -0.25.F(2.BP)
  b0.b.c := -0.125.F(3.BP)

  val b1 = Wire(new Bundle2)
  b1.a := -0.0625.F(3.BP)
  b1.b.c := -0.03125.F(4.BP)

  val b2 = Wire(new Bundle3)
  b2.a := -0.015625.F(5.BP)
  b2.b.c := -0.0078125.F(6.BP)

  val b3 = Wire(new Bundle3)
  b3.a := -0.0078125.F(7.BP)
  b3.b.c := -0.00390625.F(8.BP)

  val o1 = Mux1H(Seq(
    false.B -> b0,
    false.B -> b1,
    true.B -> b2,
    false.B -> b3
  ))

  stop()
}


