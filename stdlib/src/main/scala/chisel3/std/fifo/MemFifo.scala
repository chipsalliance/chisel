// Author: Martin Schoeberl (martin@jopdesign.com)
// License: this code is released into the public domain, see README.md and http://unlicense.org/

package chisel3.std.fifo

import chisel3._
import chisel3.util._

/**
  * FIFO with memory and read and write pointers.
  * Extra shadow register to handle the one cycle latency of the synchronous memory.
  */
class MemFifo[T <: Data](gen: T, depth: Int) extends Fifo(gen: T, depth: Int) {

  def counter(depth: Int, incr: Bool): (UInt, UInt) = {
    val cntReg = RegInit(0.U(log2Ceil(depth).W))
    val nextVal = Mux(cntReg === (depth - 1).U, 0.U, cntReg + 1.U)
    when(incr) {
      cntReg := nextVal
    }
    (cntReg, nextVal)
  }

  val mem = SyncReadMem(depth, gen)

  val incrRead = WireInit(false.B)
  val incrWrite = WireInit(false.B)
  val (readPtr, nextRead) = counter(depth, incrRead)
  val (writePtr, nextWrite) = counter(depth, incrWrite)

  val emptyReg = RegInit(true.B)
  val fullReg = RegInit(false.B)

  val idle :: valid :: full :: Nil = Enum(3)
  val stateReg = RegInit(idle)
  val shadowReg = Reg(gen)

  when(io.enq.valid && !fullReg) {
    mem.write(writePtr, io.enq.bits)
    emptyReg := false.B
    fullReg := nextWrite === readPtr
    incrWrite := true.B
  }

  val data = mem.read(readPtr)

  // Handling of the one cycle memory latency with an additional output register
  switch(stateReg) {
    is(idle) {
      when(!emptyReg) {
        stateReg := valid
        fullReg := false.B
        emptyReg := nextRead === writePtr
        incrRead := true.B
      }
    }
    is(valid) {
      when(io.deq.ready) {
        when(!emptyReg) {
          stateReg := valid
          fullReg := false.B
          emptyReg := nextRead === writePtr
          incrRead := true.B
        }.otherwise {
          stateReg := idle
        }
      }.otherwise {
        shadowReg := data
        stateReg := full
      }

    }
    is(full) {
      when(io.deq.ready) {
        when(!emptyReg) {
          stateReg := valid
          fullReg := false.B
          emptyReg := nextRead === writePtr
          incrRead := true.B
        }.otherwise {
          stateReg := idle
        }

      }
    }
  }

  io.deq.bits := Mux(stateReg === valid, data, shadowReg)
  io.enq.ready := !fullReg
  io.deq.valid := stateReg === valid || stateReg === full
}
