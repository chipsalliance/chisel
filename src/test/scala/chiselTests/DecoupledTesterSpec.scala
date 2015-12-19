package chiselTests

import Chisel._

import Chisel.testers.DecoupledTester
/**
 * Created by chick on 12/18/15.
 */

class SimpleAdder extends Module {
  val io = new Bundle {
    val a = UInt(INPUT, width=16)
    val b = UInt(INPUT, width=16)
    val c = UInt(OUTPUT, width=16)

    c := a + b
  }
}

class DecoupledTesterSpec extends ChiselFlatSpec {
  elaborate {
    new DecoupledTester {
      val device_under_test = new SimpleAdder()
      finish()

      "A DecoupledTester" should "parse identify all the io ports of a Module" in {
        assert(io_info.dut_inputs.size == 2)
        assert(io_info.dut_outputs.size == 1)

        val io = device_under_test.io
        for(port <- List(io.a, io.b, io.c)){
          assert(io_info.port_to_name.contains(port))
          println(io_info.port_to_name(port))
        }
      }
    }
  }
}
