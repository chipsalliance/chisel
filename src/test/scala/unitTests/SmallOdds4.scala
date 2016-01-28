// See LICENSE for license details.

package unitTests

import Chisel._
import Chisel.testers.OrderedDecoupledHWIOTester

import chiselTests.ChiselFlatSpec

import scala.util.Random

class SmallOdds4(filter_width: Int) extends Module {

  class FilterIO extends Bundle {
    val in = new DeqIO(UInt(width = filter_width))
    val out = new EnqIO(UInt(width = filter_width))
  }

  class Filter(isOk: UInt => Bool) extends Module {
    val io = new FilterIO

    io.in.ready := io.out.ready
    io.out.bits := io.in.bits

    io.out.valid := io.out.ready && io.in.valid && isOk(io.in.bits)
  }

  val io = new FilterIO()

  def buildFilter(): Unit = {
    val smalls = Module(new Filter(_ < UInt(10)))
    val q      = Module(new Queue(UInt(width = filter_width), entries = 1))
    val odds   = Module(new Filter((x: UInt) => (x & UInt(1)) === UInt(1)))

    io.in.ready  := smalls.io.in.ready
//TODO: Uncomment this when bulk connect is working again
//    smalls.io.in <> io.in
//    q.io.enq     <> smalls.io.out
//    odds.io.in   <> q.io.deq
//    io.out       <> odds.io.out
  }

  buildFilter()
}

class SmallOdds4Tester(width: Int) extends OrderedDecoupledHWIOTester {
  val device_under_test = Module(new SmallOdds4(filter_width = width))

//TODO: Uncomment this when bulk connect is working again
//    for (i <- 0 to 30) {
//      val num = rnd.nextInt(20)
//      println(s"random value $i $num")
//      inputEvent(device_under_test.io.in.bits -> num)
//      if (num % 2 == 1 && num < 10) {
//        outputEvent(device_under_test.io.out.bits -> num)
//      }
//    }
}

class SmallOdds4TesterSpec extends ChiselFlatSpec {
  val testWidth = 32
  "a small odds filters" should "take a stream of UInt and only pass along the odd ones < 10" in {
    assert(execute {
      new SmallOdds4Tester(testWidth)
    })
  }
}


