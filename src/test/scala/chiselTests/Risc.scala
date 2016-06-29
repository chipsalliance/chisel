// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.util._

class Risc extends Module {
  val io = new Bundle {
    val isWr   = Bool(INPUT)
    val wrAddr = UInt(INPUT, 8)
    val wrData = Bits(INPUT, 32)
    val boot   = Bool(INPUT)
    val valid  = Bool(OUTPUT)
    val out    = Bits(OUTPUT, 32)
  }
  val memSize = 256
  val file = Mem(memSize, Bits(width = 32))
  val code = Mem(memSize, Bits(width = 32))
  val pc   = Reg(init=UInt(0, 8))

  val add_op :: imm_op :: Nil = Enum(Bits(width = 8), 2)

  val inst = code(pc)
  val op   = inst(31,24)
  val rci  = inst(23,16)
  val rai  = inst(15, 8)
  val rbi  = inst( 7, 0)

  val ra = Mux(rai === Bits(0), Bits(0), file(rai))
  val rb = Mux(rbi === Bits(0), Bits(0), file(rbi))
  val rc = Wire(Bits(width = 32))

  io.valid := Bool(false)
  io.out   := Bits(0)
  rc       := Bits(0)

  when (io.isWr) {
    code(io.wrAddr) := io.wrData
  } .elsewhen (io.boot) {
    pc := UInt(0)
  } .otherwise {
    switch(op) {
      is(add_op) { rc := ra +% rb }
      is(imm_op) { rc := (rai << 8) | rbi }
    }
    io.out := rc
    when (rci === UInt(255)) {
      io.valid := Bool(true)
    } .otherwise {
      file(rci) := rc
    }
    pc := pc +% UInt(1)
  }
}

/*
class RiscTester(c: Risc) extends Tester(c) {
  def wr(addr: BigInt, data: BigInt)  = {
    poke(c.io.isWr,   1)
    poke(c.io.wrAddr, addr)
    poke(c.io.wrData, data)
    step(1)
  }
  def boot()  = {
    poke(c.io.isWr, 0)
    poke(c.io.boot, 1)
    step(1)
  }
  def tick(isBoot: Boolean)  = {
    if (isBoot)
      poke(c.io.boot, 0)
    step(1)
  }
  def I (op: UInt, rc: Int, ra: Int, rb: Int) = {
    // val cr = Cat(op, UInt(rc, 8), UInt(ra, 8), UInt(rb, 8)).litValue()
    val cr = op.litValue() << 24 | rc << 16 | ra << 8 | rb
    println("I = " + cr)    // scalastyle:ignore regex
    cr
  }

  val app  = Array(I(c.imm_op,   1, 0, 1), // r1 <- 1
                   I(c.add_op,   1, 1, 1), // r1 <- r1 + r1
                   I(c.add_op,   1, 1, 1), // r1 <- r1 + r1
                   I(c.add_op, 255, 1, 0)) // rh <- r1
  wr(0, 0) // skip reset
  for (addr <- 0 until app.length)
    wr(addr, app(addr))
  def dump(k: Int) {
    println("K = " + k)  // scalastyle:ignore regex
    peek(c.ra)
    peek(c.rb)
    peek(c.rc)
    peek(c.io.out)
    peek(c.pc)
    peek(c.inst)
    peek(c.op)
    peek(c.rci)
    peek(c.rai)
    peek(c.rbi)
    peekAt(c.file, 1)
  }
  boot()
  dump(0)
  var k = 0
  do {
    tick(k == 0); k += 1
    dump(k)
  } while (!(peek(c.io.valid) == 1 || k > 10))
  expect(k <= 10, "TIME LIMIT")
  expect(c.io.out, 4)
}
*/

class RiscSpec extends ChiselPropSpec {

  property("Risc should elaborate") {
    elaborate { new Risc }
  }

  ignore("RiscTester should return the correct result") { }
}
