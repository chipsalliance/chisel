// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
import chisel3.util.{Mux1H, UIntToOH}
import circt.stage.ChiselStage.emitCHIRRTL
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.should.Matchers

class OneHotMuxSpec extends AnyFreeSpec with Matchers with ChiselSim {
  "simple one hot mux with uint should work" in {
    simulate(new SimpleOneHotTester)(RunUntilFinished(3))
  }
  "simple one hot mux with sint should work" in {
    simulate(new SIntOneHotTester)(RunUntilFinished(3))
  }
  "simple one hot mux with all same parameterized sint values should work" in {
    simulate(new ParameterizedOneHotTester)(RunUntilFinished(3))
  }
  "UIntToOH with output width greater than 2^(input width)" in {
    simulate(new UIntToOHTester)(RunUntilFinished(3))
  }
  "UIntToOH should accept width of zero" in {
    simulate(new ZeroWidthOHTester)(RunUntilFinished(3))
  }
  "Mux1H should give a decent error when given an empty Seq" in {
    val e = intercept[IllegalArgumentException] {
      Mux1H(Seq.empty[(Bool, UInt)])
    }
    e.getMessage should include("Mux1H must have a non-empty argument")
  }
  "Mux1H should give a error when given different size Seqs" in {
    val e = intercept[ChiselException] {
      emitCHIRRTL(
        new RawModule {
          Mux1H(Seq(true.B, false.B), Seq(1.U, 2.U, 3.U))
        },
        args = Array("--throw-on-first-error")
      )
    }
    e.getMessage should include("OneHotMuxSpec.scala") // Make sure source locator comes from this file
    e.getMessage should include("Mux1H: input Seqs must have the same length, got sel 2 and in 3")
  }
  // The input bitvector is sign extended to the width of the sequence
  "Mux1H should NOT error when given mismatched selector width and Seq size" in {
    emitCHIRRTL(new RawModule {
      Mux1H("b10".U(2.W), Seq(1.U, 2.U, 3.U))
    })
  }
}

class SimpleOneHotTester extends Module {
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

class SIntOneHotTester extends Module {
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

class ParameterizedOneHotTester extends Module {
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

class UIntToOHTester extends Module {
  val out = UIntToOH(1.U, 3)
  require(out.getWidth == 3)
  assert(out === 2.U)

  val out2 = UIntToOH(0.U, 1)
  require(out2.getWidth == 1)
  assert(out2 === 1.U)

  stop()
}

class ZeroWidthOHTester extends Module {
  val out = UIntToOH(0.U, 0)
  assert(out === 0.U(0.W))
  stop()
}
