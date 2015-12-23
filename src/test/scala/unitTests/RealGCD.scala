package unitTests

import Chisel._
import Chisel.testers.{DecoupledTester, UnitTester}

class RealGCDInput extends Bundle {
  val a = Bits(width = 16)
  val b = Bits(width = 16)
}

class RealGCD extends Module {
  val io  = new Bundle {
    val in  = Decoupled(new RealGCDInput()).flip()
    val out = Decoupled(UInt(width = 16))
  }

  val x = Reg(UInt())
  val y = Reg(UInt())
  val p = Reg(init=Bool(false))

  io.in.ready := !p

  when (io.in.valid && !p) {
    x := io.in.bits.a
    y := io.in.bits.b
    p := Bool(true)
  }

  when (p) {
    when (x > y)  { x := y; y := x }
      .otherwise    { y := y - x }
  }

  io.out.bits  := x
  io.out.valid := y === Bits(0) && p
  when (io.out.valid) {
    p := Bool(false)
  }
}

class DecoupledRealGCDTester extends DecoupledTester {
  val device_under_test = Module(new RealGCD)
  val c = device_under_test // alias for dut

  for(x <- 0 until 9) {
    event(
      Array(
        c.io.in.bits.a -> 14,
        c.io.in.bits.b -> 35
      ),
      Array(c.io.out.bits -> 7)
    )
  }
  finish()
  io_info.show_ports("".r)
}

class RealGCDTests extends UnitTester {
  val c = Module( new RealGCD )

  def compute_gcd_results_and_cycles(a: Int, b: Int, depth: Int = 1): Tuple2[Int, Int] = {
    if(b == 0) (a, depth)
    else compute_gcd_results_and_cycles(b, a%b, depth+1 )
  }

  val inputs = List( (48, 32), (7, 3), (100, 10) )
  val outputs = List( 16, 1, 10)

  for( (input_1, input_2) <- inputs) {
    val (output, cycles) = compute_gcd_results_and_cycles(input_1, input_2)

    poke(c.io.in.bits.a, input_1)
    poke(c.io.in.bits.b, input_2)
    poke(c.io.in.valid,  1)

    step(1)
    expect(c.io.in.ready, 1)
    poke(c.io.in.valid, 0)
    step(1)

    step(cycles-2)
    expect(c.io.out.bits, output)
  }

  //  var i = 0
  //  do {
  //    var transfer = false
  //    do {
  //      poke(c.io.in.bits.a, inputs(i)._1)
  //      poke(c.io.in.bits.b, inputs(i)._2)
  //      poke(c.io.in.valid,  1)
  //      transfer = (peek(c.io.in.ready) == 1)
  //      step(1)
  //    } while (t < 100 && !transfer)
  //
  //    do {
  //      poke(c.io.in.valid, 0)
  //      step(1)
  //    } while (t < 100 && (peek(c.io.out.valid) == 0))
  //
  //    expect(c.io.out.bits, outputs(i))
  //    i += 1;
  //  } while (t < 100 && i < 3)
  //  if (t >= 100) ok = false

  install(c)
}
