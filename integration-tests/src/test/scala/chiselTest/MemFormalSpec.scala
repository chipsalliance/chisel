// SPDX-License-Identifier: Apache-2.0

package chiselTest

import chisel3._
import chisel3.experimental._
import chiseltest._
import chiseltest.formal._
import firrtl.annotations.MemoryArrayInitAnnotation
import org.scalatest.flatspec.AnyFlatSpec

class MemFormalSpec extends AnyFlatSpec with ChiselScalatestTester with Formal {
  behavior of "SyncReadMem read enable"

  private def check(mod: Boolean => ReadEnTestModule, alwaysEnabeld: Boolean = false): Unit = {
    // we first check that the read is enabled when it should be
    verify(mod(true), Seq(BoundedCheck(4)))
    if(!alwaysEnabeld) {
      // now we check that it is disabled, when it should be
      // however, note that this check is not exhaustive/complete!
      assertThrows[FailedBoundedCheckException] {
        verify(mod(false), Seq(BoundedCheck(4)))
      }
    }
  }

  it should "always be true when calling read(addr)" in {
    check(new ReadEnTestModule(_) { out := mem.read(addr) }, true)
  }

  it should "always be true when calling read(addr, true.B)" in {
    check(new ReadEnTestModule(_) { out := mem.read(addr, true.B) }, true)
  }

  it should "always be false when calling read(addr, false.B)" in {
    check(new ReadEnTestModule(_) {
      out := mem.read(addr, false.B)
      shouldRead := false.B
      shouldNotRead := true.B
    })
  }

  it should "be enabled by iff en=1 when calling read(addr, en)" in {
    check(new ReadEnTestModule(_) {
      val en = IO(Input(Bool()))
      out := mem.read(addr, en)
      shouldRead := en
      shouldNotRead := !en
    })
  }
}

abstract class ReadEnTestModule(testShouldRead: Boolean) extends Module {
  val addr = IO(Input(UInt(5.W)))
  val out = IO(Output(UInt(8.W)))
  out := DontCare
  // these can be overwritten by the concrete test
  val shouldRead = WireInit(true.B)
  val shouldNotRead = WireInit(false.B)

  // we initialize the memory, so that the output should always equivalent to the read address
  val mem = SyncReadMem(32, chiselTypeOf(out))
  annotate(new ChiselAnnotation {
    override def toFirrtl = MemoryArrayInitAnnotation(mem.toTarget, values = Seq.tabulate(32)(BigInt(_)))
  })

  // the first cycle after reset, the data will be arbitrary
  val firstCycle = RegNext(false.B, init=true.B)

  if(testShouldRead) {
    when(!firstCycle && RegNext(shouldRead)) {
      verification.assert(out === RegNext(addr))
    }
  } else {
    when(!firstCycle && RegNext(shouldNotRead)) {
      // this can only fail if the read enable is false and an arbitrary value is provided
      // note that this test is not complete!!
      verification.assert(out === 200.U)
    }
  }
}
