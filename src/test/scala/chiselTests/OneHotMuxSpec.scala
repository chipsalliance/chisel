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
    Driver.execute(Array.empty[String], () => new SIntOneHot) match {
      case result: ChiselExecutionSuccess =>
        println(s"SIntOneHot\n${result.emitted}")
    }
    assertTesterPasses(new SIntOneHotTester)
  }
  "simple one hot mux with fixed point should work" in {
    Driver.execute(Array.empty[String], () => new FixedPointOneHot) match {
      case result: ChiselExecutionSuccess =>
        println(s"FixedPointOneHot\n${result.emitted}")
    }
    assertTesterPasses(new FixedPointOneHotTester)
  }
  "simple one hot mux with all same fixed point should work" in {
    Driver.execute(Array.empty[String], () => new AllSameFixedPointOneHot) match {
      case result: ChiselExecutionSuccess =>
        println(s"AllSameFixedPointOneHot\n${result.emitted}")
    }
    assertTesterPasses(new AllSameFixedPointOneHotTester)
  }
  "simple one hot mux with all same parameterized sint values should work" in {
    val values: Seq[SInt] = Seq((-3).S, (-5).S, (-7).S, (-11).S)
    Driver.execute(Array.empty[String], () => new ParameterizedOneHot(values, SInt(8.W))) match {
      case result: ChiselExecutionSuccess =>
        println(s"ParameterizedOneHot\n${result.emitted}")
    }
    assertTesterPasses(new ParameterizedOneHotTester(values, SInt(8.W)))
  }
  "simple one hot mux with all same parameterized aggregates containing fixed values should work" in {
    Driver.execute(Array("--no-run-firrtl"), () => new ParameterizedAggregateOneHotTester) match {
      case result: ChiselExecutionSuccess =>
        println(s"ParameterizedOneHot\n${result.emitted}")
    }
    assertTesterPasses(new ParameterizedAggregateOneHotTester)
  }
}

class SimpleOneHotTester extends BasicTester {
  val dut = Module(new SimpleOneHot)
  dut.io.selectors(0) := false.B
  dut.io.selectors(1) := false.B
  dut.io.selectors(2) := true.B
  dut.io.selectors(3) := false.B

  assert(dut.io.out === 8.U)

  stop()
}

class SimpleOneHot extends Module {
  val io = IO(new Bundle {
    val selectors = Input(Vec(4, Bool()))
    val out = Output(UInt(8.W))
  })

  io.out := Mux1H(Seq(
    io.selectors(0) -> 2.U,
    io.selectors(1) -> 4.U,
    io.selectors(2) -> 8.U,
    io.selectors(3) -> 11.U
  ))
}
class SIntOneHotTester extends BasicTester {
  val dut = Module(new SIntOneHot)
  dut.io.selectors(0) := false.B
  dut.io.selectors(1) := true.B
  dut.io.selectors(2) := false.B
  dut.io.selectors(3) := false.B

  printf("out is %d\n", dut.io.out)

  assert(dut.io.out === (-5).S)

  stop()
}


class SIntOneHot extends Module {
  val io = IO(new Bundle {
    val selectors = Input(Vec(4, Bool()))
    val out = Output(SInt(8.W))
  })

  io.out := Mux1H(Seq(
    io.selectors(0) -> (-3).S,
    io.selectors(1) -> (-5).S,
    io.selectors(2) -> (-7).S,
    io.selectors(3) -> (-11).S
//      io.selectors(0) -> (-1).S,
//    io.selectors(1) -> (-2).S,
//    io.selectors(2) -> (-4).S,
//    io.selectors(3) -> (-11).S
  ))
}

class FixedPointOneHotTester extends BasicTester {
  val dut = Module(new FixedPointOneHot)
  dut.io.selectors(0) := false.B
  dut.io.selectors(1) := true.B
  dut.io.selectors(2) := false.B
  dut.io.selectors(3) := false.B

  assert(dut.io.out === (-2.25).F(4.BP))

  stop()
}


class FixedPointOneHot extends Module {
  val io = IO(new Bundle {
    val selectors = Input(Vec(4, Bool()))
    val out = Output(FixedPoint(8.W, 4.BP))
  })

  io.out := Mux1H(Seq(
    io.selectors(0) -> (-1.5).F(1.BP),
    io.selectors(1) -> (-2.25).F(2.BP),
    io.selectors(2) -> (-4.125).F(3.BP),
    io.selectors(3) -> (-11.625).F(3.BP)
  ))
}

class AllSameFixedPointOneHotTester extends BasicTester {
  val dut = Module(new AllSameFixedPointOneHot)
  dut.io.selectors(0) := false.B
  dut.io.selectors(1) := true.B
  dut.io.selectors(2) := false.B
  dut.io.selectors(3) := false.B

  assert(dut.io.out === (-2.25).F(4.BP))

  stop()
}


class AllSameFixedPointOneHot extends Module {
  val io = IO(new Bundle {
    val selectors = Input(Vec(4, Bool()))
    val out = Output(FixedPoint(12.W, 4.BP))
  })

  io.out := Mux1H(Seq(
    io.selectors(0) -> (-1.5).F(12.W, 4.BP),
    io.selectors(1) -> (-2.25).F(12.W, 4.BP),
    io.selectors(2) -> (-4.125).F(12.W, 4.BP),
    io.selectors(3) -> (-11.625).F(12.W, 4.BP)
  ))
}

class ParameterizedOneHotTester[T <: Data](values: Seq[T], outGen: T) extends BasicTester {
  val dut = Module(new ParameterizedOneHot(values, outGen))
  dut.io.selectors(0) := false.B
  dut.io.selectors(1) := true.B
  dut.io.selectors(2) := false.B
  dut.io.selectors(3) := false.B

  // printf(s"out is %d\n", dut.io.out)

  assert(dut.io.out.asUInt() === values(1).asUInt())

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
    w.v(0) := Wire(d.F(4.BP))
    w.v(1) := Wire(e.F(4.BP))
    w.a.f1 := Wire(f.F(3.BP))
    w.a.f2 := Wire(g.F(5.BP))
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


