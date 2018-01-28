// See LICENSE for license details.

package chisel3

class NotLiteralException(message: String) extends Exception(message)
class LiteralTypeException(message: String) extends Exception(message)

/** Basic interfaces and implicit conversions for testers2
  */
package object testers2 {
  import chisel3.internal.firrtl.{LitArg, ULit, SLit}
  implicit class testableBits(val x: Bits) {
    def getLit(data: Bits) = data.litArg match {
      case Some(value: ULit) => value.n
      case None => throw new NotLiteralException(s"$data not a literal, cannot be used in poke")
      case Some(_) => throw new LiteralTypeException(s"$data of wrong type, cannot be used to poke Bits")
    }

    def poke(value: Bits): Unit = {
      Context().backend.poke(x, getLit(value))
    }

    def peek(): Bits = {
      // TODO: fixed width based on circuit sizing?
      Context().backend.peek(x).asUInt
    }
    def stalePeek(): Bits = {
      Context().backend.stalePeek(x).asUInt
    }

    def check(value: Bits): Unit = {
      Context().backend.check(x, getLit(value))
    }
    def staleCheck(value: Bits): Unit = {
      Context().backend.staleCheck(x, getLit(value))
    }
  }

  implicit class testableClock(val x: Clock) {
    def step(cycles: Int = 1): Unit = {
      Context().backend.step(x, cycles)
    }
  }
}
