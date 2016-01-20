// See LICENSE for license details.

package unitTests

import Chisel._
import Chisel.testers.DecoupledTester

import chiselTests.ChiselFlatSpec

import scala.util.Random

class SmallOdds4(filter_width: Int) extends Module {

  class FilterIO extends Bundle {
    val in = new DeqIO(UInt(width = filter_width))
    val out = new EnqIO(UInt(width = filter_width))
  }

  class Filter(isOk: UInt => Bool) extends Module {
    val io = new FilterIO

    //    when(isOk(io.in.bits)) {
    //      io.out.enq(io.in.deq())
    //    }
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

    smalls.io.in <> io.in
    q.io.enq     <> smalls.io.out
    odds.io.in   <> q.io.deq
    io.out       <> odds.io.out
  }

  buildFilter()
}

class SmallOdds4Tester(width: Int) extends DecoupledTester {
  val device_under_test = Module(new SmallOdds4(filter_width = width))

  Random.setSeed(0L)
  for(i <- 0 to 30) {
    val num = Random.nextInt(20)
    println(s"random value $i $num")
    inputEvent(List(device_under_test.io.in.bits -> num))
    if(num % 2 == 1 && num < 10) {
      outputEvent(List(device_under_test.io.out.bits -> num))
    }
  }
  finish()
}

class SmallOdds4TesterSpec extends ChiselFlatSpec {
  "a small odds filters" should "take a stream of UInt and only pass along the odd ones < 10" in {
    assert(execute {
      new SmallOdds4Tester(32)
    })
  }
}


