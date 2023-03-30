// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.testers.BasicTester
import chisel3.util.{Mux1H, UIntToOH}
import org.scalatest._
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.should.Matchers

class OneHotMuxSpec extends AnyFreeSpec with Matchers with ChiselRunners {
  "simple one hot mux with uint should work" in {
    assertTesterPasses(new SimpleOneHotTester)
  }
  "simple one hot mux with sint should work" in {
    assertTesterPasses(new SIntOneHotTester)
  }
  "simple one hot mux with all same parameterized sint values should work" in {
    assertTesterPasses(new ParameterizedOneHotTester)
  }
  "UIntToOH with output width greater than 2^(input width)" in {
    assertTesterPasses(new UIntToOHTester)
  }
  "UIntToOH should accept width of zero" in {
    assertTesterPasses(new ZeroWidthOHTester)
  }

}

class SimpleOneHotTester extends BasicTester {
  val out = Wire(UInt())
  out := Mux1H(
    Seq(
      false.B -> 2.U,
      false.B -> 4.U,
      true.B -> 8.U,
      false.B -> 11.U
    )
  )

  assert(out === 8.U)

  stop()
}

class SIntOneHotTester extends BasicTester {
  val out = Wire(SInt())
  out := Mux1H(
    Seq(
      false.B -> (-3).S,
      true.B -> (-5).S,
      false.B -> (-7).S,
      false.B -> (-11).S
    )
  )

  assert(out === (-5).S)

  stop()
}

class ParameterizedOneHotTester extends BasicTester {
  val values: Seq[Int] = Seq(-3, -5, -7, -11)
  for ((v, i) <- values.zipWithIndex) {
    val dut = Module(new ParameterizedOneHot(values.map(_.S), SInt(8.W)))
    dut.io.selectors := (1 << i).U(4.W).asBools

    assert(dut.io.out.asUInt === v.S(8.W).asUInt)
  }

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

class UIntToOHTester extends BasicTester {
  val out = UIntToOH(1.U, 3)
  require(out.getWidth == 3)
  assert(out === 2.U)

  val out2 = UIntToOH(0.U, 1)
  require(out2.getWidth == 1)
  assert(out2 === 1.U)

  stop()
}

class ZeroWidthOHTester extends BasicTester {
  val out = UIntToOH(0.U, 0)
  assert(out === 0.U(0.W))
  stop()
}
