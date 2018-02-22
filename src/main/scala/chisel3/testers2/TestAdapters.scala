// See LICENSE for license details.

package chisel3.testers2

import chisel3._
import chisel3.util._

// TODO get rid of this boilerplate
import chisel3.internal.firrtl.{LitArg, ULit, SLit}

package object TestAdapters {
  // TODO: clock should be optional
  class ReadyValidSource[T <: Data](x: ReadyValidIO[T], clk: Clock) {
    x.valid.weakPoke(false.B)

    // TODO: poking Bundles
    def enqueueNow(data: T): Unit = {
      x.ready.expect(true.B)
      x.bits.poke(data)
      x.valid.poke(true.B)
      fork {
        clk.step(1)
        x.valid.weakPoke(false.B)
      }
    }

    def enqueueSeq(data: Seq[T]): AbstractTesterThread = {
      fork {
        for (elt <- data) {
          while (x.ready.peek().litToBoolean == false) {
            clk.step(1)
          }
          x.bits.poke(elt)
          x.valid.poke(true.B)
          clk.step(1)
          x.valid.poke(false.B)
        }
      }
    }
  }

  class ReadyValidSink[T <: Data](x: ReadyValidIO[T], clk: Clock) {
    x.ready.weakPoke(false.B)

    // TODO: poking Bundles
    def expectDequeueNow(data: T): Unit = {
      x.valid.expect(true.B)
      x.bits.expect(data)
      x.ready.poke(true.B)
      fork {
        clk.step(1)
        x.ready.weakPoke(false.B)
      }
    }

    def expectPeekNow(data: T): Unit = {
      x.valid.expect(true.B)
      x.bits.expect(data)
    }

    def expectInvalid(): Unit = {
      x.valid.expect(false.B)
    }
  }
}
