// See LICENSE for license details.

package unitTests

import Chisel._
import Chisel.testers._
import chiselTests.ChiselFlatSpec

import scala.util.Random

object Router {
  val addressWidth    = 32
  val dataWidth       = 64
  val headerWidth     =  8
  val routeTableSize  = 15
  val numberOfOutputs =  4
}

class ReadCmd extends Bundle {
  val addr = UInt(width = Router.addressWidth)
}

class WriteCmd extends ReadCmd {
  val data = UInt(width = Router.addressWidth)
}

class Packet extends Bundle {
  val header = UInt(width = Router.headerWidth)
  val body   = Bits(width = Router.dataWidth)
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
  val read_routing_table_response  = new EnqIO(UInt(width = Router.headerWidth))
  val load_routing_table_request   = new DeqIO(new WriteCmd())
  val in                           = new DeqIO(new Packet())
  val outs                         = Vec(n, new EnqIO(new Packet()))
}

/**
  * routes packets by using their header as an index into an externally loaded and readable table,
  * The number of addresses recognized does not need to match the number of outputs
  */
class Router extends Module {
  val depth = Router.routeTableSize
  val n     = Router.numberOfOutputs
  val io    = new RouterIO(n)
  val tbl   = Mem(depth, UInt(width = BigInt(n).bitLength))

  when(reset) {
    tbl.indices.foreach { index =>
      tbl(index) := UInt(0, width = Router.addressWidth)
    }
  }

  io.read_routing_table_request.init()
  io.load_routing_table_request.init()
  io.read_routing_table_response.init()
  io.in.init()
  io.outs.foreach { out => out.init() }

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
    val idx = tbl(pkt.header(log2Up(Router.routeTableSize), 0))
//    val idx = tbl(pkt.header(3, 0))
    io.outs(idx).enq(pkt)
    printf("got packet to route header %d, data %d, being routed to out(%d) ", pkt.header, pkt.body, tbl(pkt.header))
  }
}

class RouterUnitTester(number_of_packets_to_send: Int) extends DecoupledTester {
  val device_under_test = Module(new Router)
  val c = device_under_test
  verbose = true

  Random.setSeed(0)

  def readRoutingTable(addr: Int, data: Int): Unit = {
    inputEvent(List(c.io.read_routing_table_request.bits.addr -> addr))
    outputEvent(List(c.io.read_routing_table_response.bits -> data))
  }

  def writeRoutingTable(addr: Int, data: Int): Unit = {
    inputEvent(List(
      c.io.load_routing_table_request.bits.addr -> addr,
      c.io.load_routing_table_request.bits.data -> data
    ))
  }

  def writeRoutingTableWithConfirm(addr: Int, data: Int): Unit = {
    writeRoutingTable(addr, data)
    readRoutingTable(addr, data)
  }

  def routePacket(header: Int, body: Int, routed_to: Int): Unit = {
    inputEvent(List(c.io.in.bits.header -> header, c.io.in.bits.body -> body))
    outputEvent(List(c.io.outs(routed_to).bits.body -> body))
    logScala(s"rout_packet $header $body should go to out($routed_to)")
  }

  readRoutingTable(0, 0)              // confirm we initialized the routing table

  for(i <- 0 until Router.numberOfOutputs) {
    writeRoutingTableWithConfirm(i, (i + 1) % Router.numberOfOutputs) // load routing table, confirm each write as built
  }
  for(i <- Router.numberOfOutputs - 1 to 0 by -1) {
    readRoutingTable(i, (i + 1) % Router.numberOfOutputs) // check them in reverse order just for fun
  }

  // send some regular packets
  for(i <- 0 until Router.numberOfOutputs) {
    routePacket(i, i * 3, (i + 1) % 4)
  }

  // generate a new routing table
  val new_routing_table = Array.tabulate(Router.routeTableSize) { _ =>
    scala.util.Random.nextInt(Router.numberOfOutputs)
  }

  // load a new routing table
  for((destination, index) <- new_routing_table.zipWithIndex) {
    writeRoutingTable(index, destination)
  }

  // send a bunch of packets, with random values
  for(i <- 0 to number_of_packets_to_send) {
    val data = Random.nextInt(Int.MaxValue-1)
    routePacket(i % Router.routeTableSize, data, new_routing_table(i % Router.routeTableSize))
  }

  finish(show_io_table = true)
}

class RouterUnitTesterSpec extends ChiselFlatSpec {
  val number_of_packets = 20
  "a router" should "can have it's rout table loaded and changed and route a bunch of packets" in {
    assert(execute {
      new RouterUnitTester(number_of_packets)
    })
  }
}
