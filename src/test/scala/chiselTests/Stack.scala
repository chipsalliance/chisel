// See LICENSE for license details.

package chiselTests

import scala.collection.mutable.Stack

import chisel3._
import chisel3.util._

class ChiselStack(val depth: Int) extends Module {
  val io = IO(new Bundle {
    val push    = Input(Bool())
    val pop     = Input(Bool())
    val en      = Input(Bool())
    val dataIn  = Input(UInt(32.W))
    val dataOut = Output(UInt(32.W))
  })

  val stack_mem = Mem(depth, UInt(32.W))
  val sp        = RegInit(0.U(log2Ceil(depth+1).W))
  val out       = RegInit(0.U(32.W))

  when (io.en) {
    when(io.push && (sp < depth.asUInt)) {
      stack_mem(sp) := io.dataIn
      sp := sp +% 1.U
    } .elsewhen(io.pop && (sp > 0.U)) {
      sp := sp -% 1.U
    }
    when (sp > 0.U) {
      out := stack_mem(sp -% 1.U)
    }
  }
  io.dataOut := out
}

/*
class StackTester(c: Stack) extends Tester(c) {
  var nxtDataOut = 0
  var dataOut = 0
  val stack = new ScalaStack[Int]()

  for (t <- 0 until 16) {
    val enable  = rnd.nextInt(2)
    val push    = rnd.nextInt(2)
    val pop     = rnd.nextInt(2)
    val dataIn  = rnd.nextInt(256)

    if (enable == 1) {
      dataOut = nxtDataOut
      if (push == 1 && stack.length < c.depth) {
        stack.push(dataIn)
      } else if (pop == 1 && stack.length > 0) {
        stack.pop()
      }
      if (stack.length > 0) {
        nxtDataOut = stack.top
      }
    }

    poke(c.io.pop,    pop)
    poke(c.io.push,   push)
    poke(c.io.en,     enable)
    poke(c.io.dataIn, dataIn)
    step(1)
    expect(c.io.dataOut, dataOut)
  }
}
*/

class StackSpec extends ChiselPropSpec {

  property("Stack should elaborate") {
    elaborate { new ChiselStack(2) }
  }

  ignore("StackTester should return the correct result") { }
}
