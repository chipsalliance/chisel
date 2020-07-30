// See LICENSE for license details.

package chiselTests.experimental.verification

import chisel3._
import chisel3.experimental.{verification => formal}
import chisel3.stage.ChiselStage
import chiselTests.ChiselPropSpec

class VerificationModule extends Module {
  val io = IO(new Bundle{
    val in = Input(UInt(8.W))
    val out = Output(UInt(8.W))
  })
  io.out := io.in
  formal.cover(io.in === 3.U)
  when (io.in === 3.U) {
    formal.assume(io.in =/= 2.U)
    formal.assert(io.out === io.in)
  }
}

class VerificationSpec extends ChiselPropSpec {

  def assertContains[T](s: Seq[T], x: T): Unit = {
    val containsLine = s.map(_ == x).reduce(_ || _)
    assert(containsLine, s"\n  $x\nwas not found in`\n  ${s.mkString("\n  ")}``")
  }

  property("basic equality check should work") {
    val stage = new ChiselStage
    val fir = stage.emitFirrtl(new VerificationModule)
    val lines = fir.split("\n").map(_.trim)
    assertContains(lines, "cover(clock, _T, UInt<1>(\"h1\"), \"\") @[VerificationSpec.scala 16:15]")
    assertContains(lines, "assume(clock, _T_2, UInt<1>(\"h1\"), \"\") @[VerificationSpec.scala 18:18]")
    assertContains(lines, "assert(clock, _T_3, UInt<1>(\"h1\"), \"\") @[VerificationSpec.scala 19:18]")
  }
}
