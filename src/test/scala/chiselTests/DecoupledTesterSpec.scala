package chiselTests

import Chisel._

import Chisel.testers.DecoupledTester
/**
 * Created by chick on 12/18/15.
 */

class DecoupledAdderIO extends Bundle {
  val a = UInt(INPUT, width=16)
  val b = UInt(INPUT, width=16)
}

class DecoupledAdder extends Module {
  val io = new Bundle {
    val in  = Decoupled(new DecoupledAdderIO).flip()
    val out = Valid(UInt(INPUT, width = 16))
  }
  io.out.bits := io.in.bits.a + io.in.bits.b
}

class DecoupledTesterSpec extends ChiselFlatSpec {
  elaborate {
    new DecoupledTester {
      val device_under_test = new DecoupledAdder()

      event(
      Array(device_under_test.io.in.bits.a -> 4, device_under_test.io.in.bits.b -> 7),
      Array(device_under_test.io.out.bits -> 11)
      )
      finish()

      "A DecoupledTester" should "parse identify all the io ports of a Module" in {
//        assert(io_info.dut_inputs.size == 2)
//        assert(io_info.dut_outputs.size == 1)

        val dut_io = device_under_test.io
        for(port <- List(dut_io.in.bits.a, dut_io.in.bits.b, dut_io.out.bits)){
          assert(io_info.port_to_name.contains(port))
          println(io_info.port_to_name(port))
        }

        io_info.show_ports(".*".r)
      }
      it should "identify the decoupled interfaces" in {
        assert(io_info.find_parent_decoupled_port_name("in.bits") == Some("in"))
        assert(io_info.find_parent_decoupled_port_name("in.bits.a") == Some("in"))
        assert(io_info.find_parent_decoupled_port_name("in.bits.b") == Some("in"))
        assert(io_info.find_parent_valid_port_name("out.bits") == Some("out"))

      }
    }
  }
}
