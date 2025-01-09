package chiselTests.simulator

import chisel3._

class OptionalIOModule(n: Int) extends Module {
  val a, b, c = IO(Input(Bool()))
  val d, e, f = IO(Input(Bool()))
  val foo, bar = IO(Input(UInt(n.W)))
  val emptyBundle = IO(Input(new Bundle {}))
  val bundle = IO(Input(new Bundle { val x = foo.cloneType }))
  val out = IO(Output(UInt(n.W)))

  val myReg = RegInit(0.U(n.W))

  out := myReg

  when(a && b && c) {
    myReg := foo
  }
  when(d && e && f) {
    myReg := bar
  }
}
