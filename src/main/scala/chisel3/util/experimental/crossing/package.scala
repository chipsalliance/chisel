package chisel3.util.experimental

import chisel3._
import chisel3.experimental.requireIsHardware
import chisel3.util.DecoupledIO

package object crossing {

  implicit class DecoupledToAsyncIO[T <: DecoupledIO[Data]](target: T) {
    def asyncDequeue(clock: Clock, reset: Reset): AsyncDequeueIO[T] = {
      requireIsHardware(target)
      requireIsHardware(clock)
      requireIsHardware(reset)
      require(target.valid.direction == ActualDirection.Output)
      val io = Wire(new AsyncDequeueIO(target))
      io.clock := clock
      io.reset := clock
      io.sink <> target
      io
    }

    def asyncEnqueue(clock: Clock, reset: Reset): AsyncEnqueueIO[T] = {
      requireIsHardware(target)
      requireIsHardware(clock)
      requireIsHardware(reset)
      require(target.valid.direction == ActualDirection.Input)
      val io = Wire(new AsyncEnqueueIO(target))
      io.clock := clock
      io.reset := clock
      io.source <> target
      io
    }
  }

  private[crossing] def grayCounter(bits: Int, increment: Bool = true.B, clear: Bool = false.B, name: String = "binary"): UInt = {
    val incremented = Wire(UInt(bits.W))
    val binary = RegNext(next = incremented, init = 0.U).suggestName(name)
    incremented := Mux(clear, 0.U, binary + increment.asUInt())
    incremented ^ (incremented >> 1)
  }
  
}
