// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import circt.stage.ChiselStage

object IllegalRefSpec {
  class IllegalRefInner extends RawModule {
    val io = IO(new Bundle {
      val i = Input(Bool())
      val o = Output(Bool())
    })
    val x = io.i & io.i
    io.o := io.i
  }

  class IllegalRefOuter(useConnect: Boolean) extends RawModule {
    val io = IO(new Bundle {
      val a = Input(Bool())
      val b = Input(Bool())
      val out = Output(Bool())
    })

    val inst = Module(new IllegalRefInner)
    io.out := inst.io.o
    inst.io.i := io.a
    val x = WireInit(io.b)
    if (useConnect) {
      val z = WireInit(inst.x) // oops
    } else {
      val z = inst.x & inst.x // oops
    }
  }

  class CrossWhenConnect(useConnect: Boolean) extends RawModule {
    val io = IO(new Bundle {
      val i = Input(Bool())
      val o = Output(Bool())
    })
    private var tmp: Option[Bool] = None
    when(io.i) {
      val x = io.i & io.i
      tmp = Some(x)
    }
    if (useConnect) {
      io.o := tmp.get
    } else {
      val z = tmp.get & tmp.get
      io.o := io.i
    }
  }
}

class IllegalRefSpec extends ChiselFlatSpec with Utils {
  import IllegalRefSpec._

  val variants = Map("a connect" -> true, "an op" -> false)

  variants.foreach {
    case (k, v) =>
      s"Illegal cross-module references in ${k}" should "fail" in {
        a[ChiselException] should be thrownBy extractCause[ChiselException] {
          ChiselStage.emitCHIRRTL { new IllegalRefOuter(v) }
        }
      }

      s"Using a signal that has escaped its enclosing when scope in ${k}" should "fail" in {
        a[ChiselException] should be thrownBy extractCause[ChiselException] {
          ChiselStage.emitCHIRRTL { new CrossWhenConnect(v) }
        }
      }
  }
}
