// See LICENSE for license details.

package chiselTests

import org.scalatest._
import chisel3._
import chisel3.util._
import chisel3.testers.BasicTester

class CMemSanityTest extends BasicTester {
  val (cycle, done) = Counter(Bool(true), 4)
  val mem = Mem(8, UInt(width = 32))
  mem(UInt(3)) := UInt(123)
  when (cycle === UInt(1)) {
    chisel3.assert(mem(UInt(3)) === UInt(123))
  }
  when (done) { stop() }
}

class SMemSanityTest extends BasicTester {
  val (cycle, done) = Counter(Bool(true), 4)
  val mem = SeqMem(8, UInt(width = 32))
  mem(UInt(3)) := UInt(123)
  when (cycle === UInt(2)) {
    chisel3.assert(mem(UInt(3)) === UInt(123))
  }
  when (done) { stop() }
}

class MemSpec extends ChiselFlatSpec {

  behavior of "Mems"

  it should "support writing then reading a value from Combinational Mems" in {
    assertTesterPasses { new CMemSanityTest }
  }

  it should "support writing then reading a value from Sequential Mems" in {
    assertTesterPasses { new SMemSanityTest }
  }
}
