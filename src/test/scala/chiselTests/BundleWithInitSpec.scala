// See LICENSE for license details.

package chiselTests

import Chisel._
import Chisel.testers.BasicTester

/**
  * currrently the DeqIO and EnqIO contain initialization code, that leads to errors that look
  * like
  * BundleWithInitTester4190357705551496972.fir@14.6: [module BundleWithInitTester$Router]
  * Reference T_27 is not declared
  */
object BundleWithInitTester {
  val data_width = 32
}

class ReadCmd extends Bundle {
  val addr = UInt(width = BundleWithInitTester.data_width)
}

class RouterIO extends Bundle {
  val in_port   = new DeqIO(new ReadCmd)
  val out_port  = new EnqIO(new ReadCmd)
}

class BundleWithInitTester extends BasicTester {
  class Router extends Module {
    val io       = new RouterIO

    io.in_port.ready := Bool(true)

    when(io.in_port.valid && io.out_port.ready) {
      val out_data = Wire(new ReadCmd)

//      out_data.addr := io.in_port.deq().addr
      io.out_port.enq(io.in_port.deq())
    }
  }

  val dut = Module(new Router)
}

class BundleWithInitSpec extends ChiselFlatSpec {
  assertTesterPasses {
    new BundleWithInitTester
  }
}