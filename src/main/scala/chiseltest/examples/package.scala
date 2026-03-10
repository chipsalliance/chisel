// SPDX-License-Identifier: Apache-2.0

package chiseltest

/**
 * ChiselTest Examples and Test Utilities
 *
 * This package provides example modules and utilities for testing with the
 * ChiselTest compatibility layer on ChiselSim.
 */
package object examples {
  import chisel3._

  /**
   * A simple passthrough module for basic testing
   */
  class PassthroughModule extends Module {
    val io = IO(new Bundle {
      val in = Input(UInt(8.W))
      val out = Output(UInt(8.W))
    })
    io.out := io.in
  }

  /**
   * A simple counter module
   */
  class SimpleCounter extends Module {
    val io = IO(new Bundle {
      val en = Input(Bool())
      val out = Output(UInt(8.W))
    })

    val count = RegInit(0.U(8.W))

    when(io.en) {
      count := count + 1.U
    }

    io.out := count
  }

  /**
   * A simple register with write enable
   */
  class SimpleRegister extends Module {
    val io = IO(new Bundle {
      val data_in = Input(UInt(8.W))
      val write_en = Input(Bool())
      val data_out = Output(UInt(8.W))
    })

    val reg = RegInit(0.U(8.W))

    when(io.write_en) {
      reg := io.data_in
    }

    io.data_out := reg
  }

  /**
   * A simple FIFO-like queue
   */
  class SimpleQueue extends Module {
    val io = IO(new Bundle {
      val enq_data = Input(UInt(8.W))
      val enq_valid = Input(Bool())
      val enq_ready = Output(Bool())
      val deq_data = Output(UInt(8.W))
      val deq_valid = Output(Bool())
      val deq_ready = Input(Bool())
    })

    val storage = RegInit(0.U(8.W))
    val valid = RegInit(false.B)

    // Enqueue side
    io.enq_ready := !valid || (valid && io.deq_ready)

    when(reset.asBool) {
      valid := false.B
    }.elsewhen(io.enq_valid && io.enq_ready) {
      storage := io.enq_data
      valid := true.B
    }.elsewhen(io.deq_valid && io.deq_ready) {
      valid := false.B
    }

    // Dequeue side
    io.deq_data := storage
    io.deq_valid := valid
  }

}
