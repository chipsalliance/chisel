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
    val dataIn  = Input(UInt.width(32))
    val dataOut = Output(UInt.width(32))
  })

  val stack_mem = Mem(depth, UInt.width(32))
  val sp        = Reg(init = UInt(0, width = log2Up(depth + 1)))
  val out       = Reg(init = UInt(0, width = 32))

  when (io.en) {
    when(io.push && (sp < UInt(depth))) {
      stack_mem(sp) := io.dataIn
      sp := sp +% UInt(1)
    } .elsewhen(io.pop && (sp > UInt(0))) {
      sp := sp -% UInt(1)
    }
    when (sp > UInt(0)) {
      out := stack_mem(sp -% UInt(1))
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
