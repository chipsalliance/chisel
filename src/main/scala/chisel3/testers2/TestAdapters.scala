// See LICENSE for license details.

package chisel3.testers2

import chisel3._
import chisel3.util._

package object TestAdapters {
  implicit class testableReadyValidIO[T <: Data](val x: ReadyValidIO[T]) {
    // TODO: should default ready/valid to false?

    def enqueueNow(data: Bits, clk: Clock): Unit = {
      // TODO: need implicit clock - reflect containing module?
      // TODO: poking bundles?
      x.bits match {
        case x: Bits => x.poke(data)
      }
      x.valid.poke(true.B)
      x.ready.check(true.B)
      clk.step(1)
      x.valid.poke(false.B)
    }

    def expectDequeueNow(data: Bits, clk: Clock): Unit = {
      // TODO: need implicit clock - reflect containing module?
      // TODO: poking bundles?
      x.bits match {
        case x: Bits => x.check(data)
      }
      x.valid.check(true.B)
      x.ready.poke(true.B)
      clk.step(1)
      x.ready.poke(false.B)
    }
  }
}
