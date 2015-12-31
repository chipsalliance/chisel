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
  val busy  = Reg(init=Bool(false))
  val a_reg = Reg(init=UInt(0, width = 16))
  val b_reg = Reg(init=UInt(0, width = 16))
  val counter = Reg(init=SInt(10, width = 16))

//  when(counter > SInt(0)) {
//    printf("counter %d", counter)
//    counter := counter - SInt(1)
//  }.elsewhen(counter === SInt(0)) {
//    counter := counter - SInt(1)
//    busy := Bool(false)
//  }
  io.in.ready := !busy

  printf("in: ready %d   valid %d   a %d b %d   -- out:  ready %d  valid %d  c %d",
         io.in.ready, io.in.valid, io.in.bits.a, io.in.bits.b,
         io.out.ready, io.out.valid, io.out.bits.c)

  when(io.in.valid && !busy) {
    a_reg       := io.in.bits.a
    b_reg       := io.in.bits.b
    busy        := Bool(true)
//    counter     := SInt(10)
  }
  io.out.bits.c := a_reg + b_reg

  when(busy && io.out.ready) {
    io.out.valid  := Bool(true)
    io.in.ready   := Bool(true)
    busy          := Bool(false)
  }
}

class DecoupledAdderTests extends DecoupledTester {
  val device_under_test = Module(new NewDecoupledAdder())

  for {
    x <- 1 to 3
    y <- 3 to 4
    z = x + y
  } {
    input_event(
      Array(device_under_test.io.in.bits.a -> x, device_under_test.io.in.bits.b -> y)
    )
    output_event(
      Array(device_under_test.io.out.bits.c -> z)
    )
  }
  finish()
  io_info.show_ports("".r)
}

class DecoupledAdderTests2 extends DecoupledTester {
  val device_under_test = Module(new NewDecoupledAdder())
  val c = device_under_test

  val a_values = Vec((0 to 3).map(UInt(_)))
  val b_values = Vec((1 to 4).map(UInt(_)))

  val ti = Reg(init=UInt(0, width = 16))
  val pc = Reg(init=UInt(0, width = 16))
  val oc = Reg(init=UInt(0, width = 16))

  val in_done  = Reg(init=Bool(false))
  val out_done = Reg(init=Bool(false))

  ti := ti + UInt(1)
  when(ti >= UInt(30)) { stop() }
  when(in_done && out_done) { stop() }

  //printf("ti %d pc %d oc %d in_ready %d out_valid %d==============",
//    ti, pc, oc, c.io.in.ready, c.io.out.valid)
  when(c.io.in.ready) {
//    printf(s"pc %d a %d b %d", pc, a_values(pc), b_values(pc))
    c.io.in.bits.a := a_values(pc)
    c.io.in.bits.b := b_values(pc)
    c.io.in.valid  := Bool(true)
    pc := pc + UInt(1)
    when(pc >= UInt(a_values.length)) {
      in_done := Bool(true)
    }
  }

  val c_values = Vec((0 to 3).map(x => UInt(x+x+1)))
  c.io.out.ready := Bool(true)

  when(c.io.out.valid) {
    printf("oc %d go %d expected %d", oc, c.io.out.bits.c, c_values(oc))
    assert(c.io.out.bits.c === c_values(oc))
    c.io.out.ready := Bool(true)
    oc := oc + UInt(1)
    when(oc >= UInt(c_values.length)) {
      out_done := Bool(true)
    }
  }
//  finish()
//  io_info.show_ports("".r)
}

class DecoupledAdderTester extends ChiselFlatSpec {
  "a" should "b" in {
    assert( execute { new DecoupledAdderTests2 } )
  }
}



