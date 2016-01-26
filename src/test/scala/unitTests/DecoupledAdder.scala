// See LICENSE for license details.

package unitTests

import Chisel._
import Chisel.testers.OrderedDecoupledHWIOTester
import chiselTests.ChiselFlatSpec

/**
  * Implements an adder that used decoupledIO for both input and output
  * This adder has a manual delay that is 10 cycles per add, just
  * to exercise the ready valid stuff
  */
class SlowDecoupledAdderIn extends Bundle {
  val a = UInt(width=16)
  val b = UInt(width=16)
}

class SlowDecoupledAdderOut extends Bundle {
  val c = UInt(OUTPUT, width=16)
}

class SlowDecoupledAdder extends Module {
  val delay_value = 10
  val io = new Bundle {
    val in  = Decoupled(new SlowDecoupledAdderIn).flip()
    val out = Decoupled(new SlowDecoupledAdderOut)
  }
  val busy    = Reg(init=Bool(false))
  val a_reg   = Reg(init=UInt(0, width = 16))
  val b_reg   = Reg(init=UInt(0, width = 16))
  val wait_counter = Reg(init=UInt(0, width = 16))

  io.in.ready := !busy

  printf("in: ready %d   valid %d   a %d b %d   -- out:  ready %d  valid %d  c %d",
         io.in.ready, io.in.valid, io.in.bits.a, io.in.bits.b,
         io.out.ready, io.out.valid, io.out.bits.c)

  when(io.in.valid && !busy) {
    a_reg        := io.in.bits.a
    b_reg        := io.in.bits.b
    busy         := Bool(true)
    wait_counter := UInt(0)
  }
  when(busy) {
    when(wait_counter > UInt(delay_value)) {
      io.out.bits.c := a_reg + b_reg
    }.otherwise {
      wait_counter := wait_counter + UInt(1)
    }
  }

  io.out.valid := (io.out.bits.c === a_reg + b_reg ) && busy

  when(io.out.valid) {
    busy          := Bool(false)
  }
}

class DecoupledAdderTests extends OrderedDecoupledHWIOTester {
  val device_under_test = Module(new SlowDecoupledAdder())

  testBlock {

    for {
      x <- 0 to 4
      y <- 0 to 6 by 2
      z = x + y
    } {
      inputEvent(device_under_test.io.in.bits.a -> x, device_under_test.io.in.bits.b -> y)
      outputEvent(device_under_test.io.out.bits.c -> z)
    }
  }
}


class DecoupledAdderTester extends ChiselFlatSpec {
  "decoupled adder" should "add a bunch numbers slowly but correctly" in {
    assert( execute( new DecoupledAdderTests ) )
  }
}



