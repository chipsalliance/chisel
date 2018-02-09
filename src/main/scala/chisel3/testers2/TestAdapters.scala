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
    def enqueueNow(data: Bits): Unit = {
      x.ready.check(true.B)
      x.bits match {  // TODO get rid of this boilerplate
        case x: Bits => x.poke(data)
      }
      x.valid.poke(true.B)
      fork {
        clk.step(1)
        x.valid.weakPoke(false.B)
      }
    }

    def enqueueSeq(data: Seq[Bits]): AbstractTesterThread = {
      fork {
        for (elt <- data) {
          while (x.ready.peek().litArg.get.asInstanceOf[ULit].n != 1) {
            clk.step(1)
          }
          x.bits match {
            case x: Bits => x.poke(elt)
          }
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
    def checkDequeueNow(data: Bits): Unit = {
      x.valid.check(true.B)
      x.bits match {
        case x: Bits => x.check(data)
      }
      x.ready.poke(true.B)
      fork {
        clk.step(1)
        x.ready.weakPoke(false.B)
      }
    }

    def checkPeekNow(data: Bits): Unit = {
      x.valid.check(true.B)
      x.bits match {
        case x: Bits => x.check(data)
      }
    }

    def checkInvalid(): Unit = {
      x.valid.check(false.B)
    }
  }
}
