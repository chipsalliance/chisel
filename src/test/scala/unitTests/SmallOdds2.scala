// See LICENSE for license details.

package unitTests

import Chisel._
import Chisel.testers.DecoupledTester

import chiselTests.ChiselFlatSpec

class SmallOdds2(filter_width: Int) extends Module {

  class FilterIO extends Bundle {
    val in  = new DeqIO(UInt(width = filter_width))
    val out = new EnqIO(UInt(width = filter_width))
  }

  class Filter(isOk: UInt => Bool) extends Module {
    val io = new FilterIO

//    when(isOk(io.in.bits)) {
//      io.out.enq(io.in.deq())
//    }
    io.in.ready      := Bool(true)
    io.out.bits      := io.in.bits

    io.out.valid     := io.out.ready && io.in.valid && isOk(io.in.bits)
  }

  val io = new FilterIO()

  val smalls = Module(new Filter(_ < UInt(10)))
//  val q      = Module(new Queue(UInt(width = filter_width), entries = 1))
  val odds   = Module(new Filter((x: UInt) => (x & UInt(1)) === UInt(1)))

  smalls.io.in.valid  := io.in.valid
  smalls.io.in.bits   := io.in.bits
  smalls.io.out.ready := odds.io.in.ready

  odds.io.in.valid    := smalls.io.out.valid
  odds.io.in.bits     := smalls.io.out.bits
  odds.io.out.ready   := io.out.ready

  io.out.bits         := odds.io.out.bits
  io.out.valid        := odds.io.out.valid

  io.in.ready         := Bool(true)

  printf("smalls.out.v, r %d, %d, odds.out.v,r %d, %d", smalls.io.out.valid, smalls.io.out.ready,
    odds.io.out.valid, odds.io.out.ready)

  // bulk connect goes src -> sink
//  io.in         <> smalls.io.in
//  smalls.io.out <> q.io.deq
//  q.io.enq      <> odds.io.in
//  odds.io.out   <> io.out

  // or it goes sink <- src
//  smalls.io.in  <> io.in
//  q.io.deq      <> smalls.io.out
//  odds.io.in    <> q.io.enq
//  io.out        <> odds.io.out

  // bulk connect goes src -> sink
//  io.in         <> smalls.io.in
//  q.io.deq      <> smalls.io.out
//  odds.io.in    <> q.io.enq
//  io.out        <> odds.io.out

  // or it goes sink <- src
//  smalls.io.in  <> io.in
//  q.io.deq      <> smalls.io.out
//  odds.io.in    <> q.io.enq
//  io.out        <> odds.io.out
}

class SmallOdds2Tester(width: Int) extends DecoupledTester {
  val device_under_test = Module(new SmallOdds2(filter_width = width))
  verbose = true

  val ti = Reg(init=UInt(0, width = 32))
  ti := ti + UInt(1)
  when(ti > UInt(10)) {
    printf("XXXXXXXXXXXXXXX")
    stop()
  }


  for(i <- 0 to 30) {
    inputEvent(List(device_under_test.io.in.bits -> i))
    if(i % 2 == 1 && i < 10) {
      outputEvent(List(device_under_test.io.out.bits -> i))
    }
  }
  finish()
}

class SmallOdds2TesterSpec extends ChiselFlatSpec {
  "a small odds filters" should "take a stream of UInt and only pass along the odd ones < 10" in {
    assert(execute {
      new SmallOdds2Tester(32)
    })
  }
}


