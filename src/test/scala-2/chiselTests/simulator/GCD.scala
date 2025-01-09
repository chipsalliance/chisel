package chiselTests.simulator

import chisel3._

/** A simple module useful for testing Chisel generation and testing */
class GCD extends Module {
  val io = IO(new Bundle {
    val a = Input(UInt(32.W))
    val b = Input(UInt(32.W))
    val loadValues = Input(Bool())
    val result = Output(UInt(32.W))
    val resultIsValid = Output(Bool())
  })
  val x = Reg(UInt(32.W))
  val y = Reg(UInt(32.W))
  when(x > y) { x := x -% y }.otherwise { y := y -% x }
  when(io.loadValues) { x := io.a; y := io.b }
  io.result := x
  io.resultIsValid := y === 0.U
}
