// See LICENSE for license details.

package chisel3.testers2

import chisel3._
import chisel3.util._

package object TestAdapters {
  // TODO: clock should be optional
  class ReadyValidSource[T <: Data](x: ReadyValidIO[T], clk: Clock) {
    x.valid.poke(false.B)

    // TODO: poking Bundles
    def enqueueNow(data: Bits): Unit = {
      x.bits match {
        case x: Bits => x.poke(data)
      }
      x.valid.poke(true.B)
      x.ready.check(true.B)
      fork {
        clk.step(1)
        x.valid.poke(false.B)
      }
    }
  }

  class ReadyValidSink[T <: Data](x: ReadyValidIO[T], clk: Clock) {
    x.ready.poke(false.B)

    // TODO: poking Bundles
    def checkDequeueNow(data: Bits): Unit = {
      x.bits match {
        case x: Bits => x.check(data)
      }
      x.valid.check(true.B)
      x.ready.poke(true.B)
      fork {
        clk.step(1)
        x.ready.poke(false.B)
      }
    }

    def checkInvalid(): Unit = {
      x.valid.check(false.B)
    }
  }
}
