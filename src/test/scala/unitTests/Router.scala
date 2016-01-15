// See LICENSE for license details.

package unitTests

import Chisel._
import Chisel.testers._
import chiselTests.ChiselFlatSpec

import scala.util.Random

class ReadCmd extends Bundle {
  val addr = UInt(width = 32)
}

class WriteCmd extends ReadCmd {
  val data = UInt(width = 32)
}

class Packet extends Bundle {
  val header = UInt(width = 8)
  val body   = Bits(width = 64)
}

/**
  * This router circuit
  * It routes a packet placed on it's input to one of n output ports
  *
  * @param n is number of fanned outputs for the routed packet
  */
class RouterIO(n: Int) extends Bundle {
//  override def cloneType           = new RouterIO(n).asInstanceOf[this.type]
  val read_routing_table_request   = new DeqIO(new ReadCmd())
  val read_routing_table_response  = new EnqIO(UInt(width = 8))
  val load_routing_table_request   = new DeqIO(new WriteCmd())
  val in                           = new DeqIO(new Packet())
  val outs                         = Vec(n, new EnqIO(new Packet()))
}

class Router extends Module {
  val depth = 32
  val n     = 4
  val io    = new RouterIO(n)
  val tbl   = Mem(depth, UInt(width = BigInt(n).bitLength))

  when(reset) {
    tbl.indices.foreach { index =>
      tbl(index) := UInt(0, width = 32)
    }
  }

  io.read_routing_table_request.init()
  io.load_routing_table_request.init()
  io.read_routing_table_response.init()
  io.in.init()
  io.outs.foreach { out => out.init() }

  val ti = Reg(init=UInt(0, width = 16))
  ti := ti + UInt(1)

  printf("                    tbl: %d : %d : %d : %d",
    tbl(0),
    tbl(1),
    tbl(2),
    tbl(3)
  )

  when(io.read_routing_table_request.valid && io.read_routing_table_response.ready) {
    io.read_routing_table_response.enq(tbl(
      io.read_routing_table_request.deq().addr
    ))
  }

  when(io.load_routing_table_request.fire()) {
    val cmd = io.load_routing_table_request.deq()
    tbl(cmd.addr) := cmd.data
    printf("setting tbl(%d) to %d", cmd.addr, cmd.data)
  }

  when(io.in.fire()) {
    val pkt = io.in.deq()
    val idx = tbl(pkt.header(1, 0))
    io.outs(idx).enq(pkt)
    printf("got packet to route header %d, data %d, being routed to out(%d) ", pkt.header, pkt.body, tbl(pkt.header))
  }
}

class RouterUnitTester(number_of_packets_to_send: Int) extends DecoupledTester {
  val device_under_test = Module(new Router)
  val c = device_under_test
  verbose = true

  val ti = Reg(init=UInt(0, width = 16))
  ti := ti + UInt(1)
  when(ti >= UInt(50)) { stop() }

  printf("ti %d, read %d %d,   write %d %d   in.ready %d %d",
        ti,
        c.io.read_routing_table_request.ready,
        c.io.read_routing_table_request.valid,
        c.io.load_routing_table_request.ready,
        c.io.load_routing_table_request.valid,
        c.io.in.ready,
        c.io.in.valid
  )

  def read_routing_table(addr: Int, data: Int) = {
    inputEvent(List(c.io.read_routing_table_request.bits.addr -> addr))
    outputEvent(List(c.io.read_routing_table_response.bits -> data))
  }

  def write_routing_table(addr: Int, data: Int)  = {
    inputEvent(List(
      c.io.load_routing_table_request.bits.addr -> addr,
      c.io.load_routing_table_request.bits.data -> data
    ))
  }

  def write_routing_table_with_confirm(addr: Int, data: Int): Unit = {
    write_routing_table(addr, data)
    read_routing_table(addr, data)
  }

  def route_packet(header: Int, body: Int, routed_to: Int)  = {

    inputEvent(List(c.io.in.bits.header -> header, c.io.in.bits.body -> body))
    outputEvent(List(c.io.outs(routed_to).bits.body -> body))
    logScala(s"rout_packet $header $body should go to out($routed_to)")
  }

  read_routing_table(0, 0)                // confirm we initialized the routing table
  write_routing_table_with_confirm(0, 1)  // load a routing table, confirm each write as built
  write_routing_table_with_confirm(1, 2)
  write_routing_table_with_confirm(2, 3)
  write_routing_table_with_confirm(3, 0)
  read_routing_table(1, 2)

  // send some regular packets
  for(i <- 0 until 4) {
    route_packet(i, i*3, (i+1) % 4)
  }

  val new_routing_table = scala.util.Random.shuffle((0 until c.n).toList)

  // load a new routing table
  for((destination, index) <- new_routing_table.zipWithIndex) {
    write_routing_table(index, destination)
  }

  // send a bunch of packets, with random values
  for(i <- 0 to number_of_packets_to_send) {
    val data = Random.nextInt(1000)
    route_packet(i % 4, data, new_routing_table(i % 4))
  }

  finish(show_io_table = true)
}

class RouterUnitTesterSpec extends ChiselFlatSpec {
  "a router" should "can have it's rout table loaded and changed and route a bunch of packets" in {
    assert(execute {
      new RouterUnitTester(number_of_packets_to_send = 20)
    })
  }
}
