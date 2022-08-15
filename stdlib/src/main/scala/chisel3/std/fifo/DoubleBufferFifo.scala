// Author: Martin Schoeberl (martin@jopdesign.com)

package chisel3.std.fifo

import chisel3._
import chisel3.util._

/**
  * Double buffer FIFO.
  * Maximum throughput is one word per clock cycle.
  * Each stage has a shadow buffer to handle the downstream full.
  */
class DoubleBufferFifo[T <: Data](gen: T, depth: Int) extends Fifo(gen: T, depth: Int) {

  private class DoubleBuffer[T <: Data](gen: T) extends Module {
    val io = IO(new FifoIO(gen))

    val empty :: one :: two :: Nil = Enum(3)
    val stateReg = RegInit(empty)
    val dataReg = Reg(gen)
    val shadowReg = Reg(gen)

    switch(stateReg) {
      is(empty) {
        when(io.enq.valid) {
          stateReg := one
          dataReg := io.enq.bits
        }
      }
      is(one) {
        when(io.deq.ready && !io.enq.valid) {
          stateReg := empty
        }
        when(io.deq.ready && io.enq.valid) {
          stateReg := one
          dataReg := io.enq.bits
        }
        when(!io.deq.ready && io.enq.valid) {
          stateReg := two
          shadowReg := io.enq.bits
        }
      }
      is(two) {
        when(io.deq.ready) {
          dataReg := shadowReg
          stateReg := one
        }

      }
    }

    io.enq.ready := (stateReg === empty || stateReg === one)
    io.deq.valid := (stateReg === one || stateReg === two)
    io.deq.bits := dataReg
  }

  private val buffers = Array.fill((depth + 1) / 2) { Module(new DoubleBuffer(gen)) }

  for (i <- 0 until (depth + 1) / 2 - 1) {
    buffers(i + 1).io.enq <> buffers(i).io.deq
  }
  io.enq <> buffers(0).io.enq
  io.deq <> buffers((depth + 1) / 2 - 1).io.deq
}
