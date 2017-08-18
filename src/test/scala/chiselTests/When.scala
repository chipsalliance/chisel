// See LICENSE for license details.

package chiselTests

import org.scalatest._

import chisel3._
import chisel3.testers.BasicTester
import chisel3.util._
//import chisel3.core.ExplicitCompileOptions.Strict

class WhenTester() extends BasicTester {
  val cnt = Counter(4)
  when(true.B) { cnt.inc() }

  val out = Wire(UInt(3.W))
  when(cnt.value === 0.U) {
    out := 1.U
  } .elsewhen (cnt.value === 1.U) {
    out := 2.U
  } .elsewhen (cnt.value === 2.U) {
    out := 3.U
  } .otherwise {
    out := 0.U
  }

  assert(out === cnt.value + 1.U)

  when(cnt.value === 3.U) {
    stop()
  }
}

class OverlappedWhenTester() extends BasicTester {
  val cnt = Counter(4)
  when(true.B) { cnt.inc() }

  val out = Wire(UInt(3.W))
  when(cnt.value <= 0.U) {
    out := 1.U
  } .elsewhen (cnt.value <= 1.U) {
    out := 2.U
  } .elsewhen (cnt.value <= 2.U) {
    out := 3.U
  } .otherwise {
    out := 0.U
  }

  assert(out === cnt.value + 1.U)

  when(cnt.value === 3.U) {
    stop()
  }
}

class NoOtherwiseOverlappedWhenTester() extends BasicTester {
  val cnt = Counter(4)
  when(true.B) { cnt.inc() }

  val out = Wire(UInt(3.W))
  when(cnt.value <= 0.U) {
    out := 1.U
  } .elsewhen (cnt.value <= 1.U) {
    out := 2.U
  } .elsewhen (cnt.value <= 2.U) {
    out := 3.U
  } .elsewhen (cnt.value <= 3.U) {
    out := 0.U
  }

  assert(out === cnt.value + 1.U)

  when(cnt.value === 3.U) {
    stop()
  }
}

class WhenSpec extends ChiselFlatSpec {
  "When, elsewhen, and otherwise with orthogonal conditions" should "work" in {
    assertTesterPasses{ new WhenTester }
  }
  "When, elsewhen, and otherwise with overlapped conditions" should "work" in {
    assertTesterPasses{ new OverlappedWhenTester }
  }
  "When and elsewhen without otherwise with overlapped conditions" should "work" in {
    assertTesterPasses{ new NoOtherwiseOverlappedWhenTester }
  }
}
