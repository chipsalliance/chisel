// See LICENSE for license details.

package unitTests

import Chisel._
import Chisel.testers.SteppedHWIOTester
import chiselTests.ChiselFlatSpec

class DynamicMemorySearch(val n: Int, val w: Int) extends Module {
  val io = new Bundle {
    val isWr   = Bool(INPUT)
    val wrAddr = UInt(INPUT,  log2Up(n))
    val data   = UInt(INPUT,  w)
    val en     = Bool(INPUT)
    val target = UInt(OUTPUT, log2Up(n))
    val done   = Bool(OUTPUT)
  }
  val index  = Reg(init = UInt(0, width = log2Up(n)))
  val list   = Mem(n, UInt(width = w))
  val memVal = list(index)
  val over   = !io.en && ((memVal === io.data) || (index === UInt(n-1)))

  when(reset) {
    for(i <- 0 to n) {
      list(i) := UInt(0)
    }
  }
  when (io.isWr) {
    list(io.wrAddr) := io.data
  } .elsewhen (io.en) {
    index := UInt(0)
  } .elsewhen (over === Bool(false)) {
    index := index + UInt(1)
  }
  io.done   := over
  io.target := index
}

class DynamicMemorySearchTests(val n: Int, val w: Int) extends SteppedHWIOTester {
  val device_under_test = Module(new DynamicMemorySearch(n, w))
  val c = device_under_test

  enable_all_debug = true

  val list = Array.fill(c.n)(0)
  rnd.setSeed(0L)

  for (k <- 0 until 16) {
    // WRITE A WORD
    poke(c.io.en, 0)
    poke(c.io.isWr, 1)
    val wrAddr = rnd.nextInt(c.n - 1)
    val data = rnd.nextInt((1 << c.w) - 1) + 1 // can't be 0
    poke(c.io.wrAddr, wrAddr)
    poke(c.io.data, data)
    step(1)
    list(wrAddr) = data
    // SETUP SEARCH
    val target = if (k > 12) rnd.nextInt(1 << c.w) else data
    poke(c.io.isWr, 0)
    poke(c.io.data, target)
    poke(c.io.en, 1)
    step(1)
    poke(c.io.en, 0)
    step(1)
    val expectedIndex = if (list.contains(target)) {
      list.indexOf(target)
    } else {
      list.length - 1
    }
    step(expectedIndex)
    expect(c.io.done, 1)
    expect(c.io.target, expectedIndex)
    step(1)
  }
}

class DynamicMemorySearchTester extends ChiselFlatSpec {
  val num_elements =  8
  val width        =  4
  "a dynamic memory search" should "be able to find things that were put in memory" in {
    assert(hwTest {
      new DynamicMemorySearchTests(num_elements, width)
    })
  }
}