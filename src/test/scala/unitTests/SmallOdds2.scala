// See LICENSE for license details.

package unitTests

import Chisel._
import Chisel.testers.DecoupledTester

import chiselTests.ChiselFlatSpec

class SmallOdds1[T <: UInt](tipe: T) extends Module {

  class FilterData[T <: UInt](tipe: T) extends Bundle {
    override def cloneType: this.type = new FilterData(tipe).asInstanceOf[this.type]
    val data = tipe
  }

  class FilterIO[T <: UInt] extends Bundle {
    val in  = new DeqIO(new FilterData(tipe))
    val out = new EnqIO(new FilterData(tipe))
  }

  class Filter[T <: UInt](isOk: (T) => Bool, tipe: T) extends Module {
    val io = new FilterIO

    io.out.bits.data := io.in.bits.data
    io.out.ready     := Bool(true)
    io.out.valid     := io.out.ready && io.in.valid // && isOk(io.in.bits.data)
  }

  val io = new FilterIO()
  val smalls = Module(new Filter((x: T) => x < UInt(10), tipe))
  val q      = Module(new Queue(new FilterData(tipe), entries = 1))
  val odds   = Module(new Filter((x: T) => (x & UInt(1)) > UInt(0), tipe))

  smalls.io.in  <> io.in
  smalls.io.out <> q.io.enq
  q.io.deq      <> odds.io.in
  odds.io.out   <> io.out
}

class SmallOdds1Tester extends DecoupledTester {
  val device_under_test = Module(new SmallOdds1(UInt(width = 32)))
  verbose = true

  for(i <- 0 to 30) {
    inputEvent(List(device_under_test.io.in.bits.data -> i))
    if(i % 2 == 1 && i < 10) {
      outputEvent(List(device_under_test.io.out.bits.data -> i))
    }
  }
  finish()
}

class SmallOdds1TesterSpec extends ChiselFlatSpec {
  "a small odds filters" should "take a stream of UInt and only pass along the odd ones < 10" in {
    assert(execute {
      new SmallOdds1Tester
    })
  }
}


