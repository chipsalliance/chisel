// See LICENSE for license details.

package unitTests

import Chisel._
import Chisel.testers.OrderedDecoupledHWIOTester

import chiselTests.ChiselFlatSpec

import scala.util.Random

class SmallOdds2(filter_width: Int) extends Module {

  class FilterIO extends Bundle {
    val in = new DeqIO(UInt(width = filter_width))
    val out = new EnqIO(UInt(width = filter_width))
  }

  class Filter(isOk: UInt => Bool) extends Module {
    val io = new FilterIO

    //    when(isOk(io.in.bits)) {
    //      io.out.enq(io.in.deq())
    //    }
    io.in.ready := Bool(true)
    io.out.bits := io.in.bits

    io.out.valid := io.out.ready && io.in.valid && isOk(io.in.bits)
  }

  val io = new FilterIO()

  def buildFilter(): Unit = {
    val smalls = Module(new Filter(_ < UInt(10)))
    //  val q      = Module(new Queue(UInt(width = filter_width), entries = 1))
    val odds = Module(new Filter((x: UInt) => (x & UInt(1)) === UInt(1)))

    smalls.io.in.valid := io.in.valid
    smalls.io.in.bits := io.in.bits
    smalls.io.out.ready := odds.io.in.ready

    odds.io.in.valid := smalls.io.out.valid
    odds.io.in.bits := smalls.io.out.bits
    odds.io.out.ready := io.out.ready

    io.out.bits := odds.io.out.bits
    io.out.valid := odds.io.out.valid

    io.in.ready := Bool(true)
  }

  buildFilter()
}

class SmallOdds2Tester(width: Int) extends OrderedDecoupledHWIOTester {
  val device_under_test = Module(new SmallOdds2(filter_width = width))

  testBlock {
    rnd.setSeed(0L)
    for (i <- 0 to 30) {
      val num = rnd.nextInt(20)
      println(s"random value $i $num")
      inputEvent(device_under_test.io.in.bits -> num)
      if (num % 2 == 1 && num < 10) {
        outputEvent(device_under_test.io.out.bits -> num)
      }
    }
  }
}

class SmallOdds2TesterSpec extends ChiselFlatSpec {
  "a small odds filters" should "take a stream of UInt and only pass along the odd ones < 10" in {
    assert(execute {
      new SmallOdds2Tester(32)
    })
  }
}


