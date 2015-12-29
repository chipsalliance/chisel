package chiselTests

import Chisel._

import Chisel.testers.DecoupledTester
/**
 * Created by chick on 12/18/15.
 */

class NewDecoupledAdderIn extends Bundle {
  val a = UInt(width=16)
  val b = UInt(width=16)
}

class NewDecoupledAdderOut extends Bundle {
  val c = UInt(OUTPUT, width=16)
}

class NewDecoupledAdder extends Module {
  val io = new Bundle {
    val in  = Decoupled(new NewDecoupledAdderIn).flip()
    val out = Decoupled(new NewDecoupledAdderOut)
  }
  val ready = Reg(Bool(true))
  val busy  = Reg(Bool(false))
  val a_reg = Reg(UInt(0, width = 16))
  val b_reg = Reg(UInt(0, width = 16))

  io.in.ready := ready

  when(io.in.valid) {
    a_reg := io.in.bits.a
    b_reg := io.in.bits.b
    io.in.ready := Bool(false)
    ready := Bool(false)
    busy  := Bool(true)
  }
  when(busy && io.out.ready) {
    io.out.bits.c := a_reg + b_reg
    io.out.valid  := Bool(true)
    io.in.ready   := Bool(true)
    busy          := Bool(false)
  }
}

class DecoupledAdderTests extends DecoupledTester {
  val device_under_test = Module(new NewDecoupledAdder())

  input_event(
    Array(device_under_test.io.in.bits.a -> 4, device_under_test.io.in.bits.b -> 7)
  )
  output_event(
    Array(device_under_test.io.out.bits.c -> 11)
  )
  finish()
  io_info.show_ports("".r)
}

class DecoupledAdderTester extends ChiselFlatSpec {
  "a" should "b" in {
    assert( execute { new DecoupledAdderTests } )
  }
}



