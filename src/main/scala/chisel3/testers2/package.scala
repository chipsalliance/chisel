// See LICENSE for license details.

package chisel3

class NotLiteralException(message: String) extends Exception(message)
class LiteralTypeException(message: String) extends Exception(message)

/** Basic interfaces and implicit conversions for testers2
  */
package object testers2 {
  import chisel3.internal.firrtl.{LitArg, ULit, SLit}
  implicit class testableData[T <: Data](x: T) {
    protected def litToBigInt(lit: Bits): BigInt = lit.litArg match {
      case Some(value: ULit) => value.n
      case Some(value: SLit) => value.n
      case Some(_) => throw new LiteralTypeException(s"$lit of wrong type, cannot be used to poke Bits")
      case None => throw new NotLiteralException(s"$lit not a literal, cannot be used in poke")
    }

    protected def pokeWithPriority(value: T, priority: Int): Unit = (x, value) match {
      case (x: Bool, value: Bool) => Context().backend.pokeBits(x, litToBigInt(value), priority)
      case (x: Bool, value: Bits) => throw new LiteralTypeException(s"can only poke signals of type Bool with Bool value")
      case (x: Bits, value: UInt) => Context().backend.pokeBits(x, litToBigInt(value), priority)
      case (x: SInt, value: SInt) => Context().backend.pokeBits(x, litToBigInt(value), priority)
      case (x: Bits, value: SInt) => throw new LiteralTypeException(s"can only poke SInt value into signals of type SInt")
      case x => throw new LiteralTypeException(s"don't know how to poke $x")
      // TODO: aggregate types
    }

    def poke(value: T): Unit = pokeWithPriority(value, 0)
    def weakPoke(value: T): Unit = pokeWithPriority(value, 1)

    protected def peekWithStale(stale: Boolean): T = x match {
      case (x: Bool) => Context().backend.peekBits(x, stale) match {
        case x: BigInt if x == 0 => false.B.asInstanceOf[T]
        case x: BigInt if x == 1 => true.B.asInstanceOf[T]
        case x => throw new LiteralTypeException(s"peeked Bool with value $x not 0 or 1")
      }
      case (x: UInt) => Context().backend.peekBits(x, stale).asUInt(x.width).asInstanceOf[T]
      case (x: SInt) => Context().backend.peekBits(x, stale).asSInt(x.width).asInstanceOf[T]
      case x => throw new LiteralTypeException(s"don't know how to peek $x")
    }

    def peek(): T = peekWithStale(false)
    def stalePeek(): T = peekWithStale(true)

    protected def expectWithStale(value: T, stale: Boolean): Unit = (x, value) match {
      case (x: Bool, value: Bool) => Context().backend.expectBits(x, litToBigInt(value), stale)
      case (x: Bool, value: Bits) => throw new LiteralTypeException(s"can only expect signals of type Bool with Bool value")
      case (x: Bits, value: UInt) => Context().backend.expectBits(x, litToBigInt(value), stale)
      case (x: SInt, value: SInt) => Context().backend.expectBits(x, litToBigInt(value), stale)
      case (x: Bits, value: SInt) => throw new LiteralTypeException(s"can only expect SInt value from signals of type SInt")
      case x => throw new LiteralTypeException(s"don't know how to expect $x")
      // TODO: aggregate types
    }

    def expect(value: T): Unit = expectWithStale(value, false)
    def staleExpect(value: T): Unit = expectWithStale(value, true)
  }

  implicit class testableClock(val x: Clock) {
    def step(cycles: Int = 1): Unit = {
      Context().backend.step(x, cycles)
    }
  }

  def fork(runnable: => Unit): AbstractTesterThread = {
    Context().backend.fork(runnable)
  }

  def join(thread: AbstractTesterThread) = {
    Context().backend.join(thread)
  }
}
