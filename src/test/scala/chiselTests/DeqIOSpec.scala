// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.testers.BasicTester
import chisel3.util._

/**
  * Created by chick on 2/8/16.
  */
class UsesDeqIOInfo extends Bundle {
  val test_width = 32

  val info_data = UInt.width(test_width)
}

class UsesDeqIO extends Module {
  val io = IO(new Bundle {
    val in = DeqIO(new UsesDeqIOInfo)
    val out = EnqIO(new UsesDeqIOInfo)
  })
}

class DeqIOSpec extends ChiselFlatSpec {
  runTester {
    new BasicTester {
      val dut = new UsesDeqIO
/*
      "DeqIO" should "set the direction of it's parameter to INPUT" in {
        assert(dut.io.in.bits.info_data.dir === INPUT)
      }
      "DeqIO" should "create a valid input and ready output" in {
        assert(dut.io.in.valid.dir === INPUT)
        assert(dut.io.in.ready.dir === OUTPUT)
      }
      "EnqIO" should "set the  direction of it's parameter OUTPUT" in {
        assert(dut.io.out.bits.info_data.dir === OUTPUT)
      }
      "EnqIO" should "create a valid input and ready output" in {
        assert(dut.io.out.valid.dir === OUTPUT)
        assert(dut.io.out.ready.dir === INPUT)
      }

      val in_clone = dut.io.in.cloneType
      val out_clone = dut.io.out.cloneType

      "A deqIO device" should "clone itself with it's directions intact" in {
        assert(dut.io.in.bits.info_data.dir == in_clone.bits.info_data.dir)
        assert(dut.io.in.ready.dir == in_clone.ready.dir)
        assert(dut.io.in.valid.dir == in_clone.valid.dir)
      }

      "A enqIO device" should "clone itself with it's directions intact" in {
        assert(dut.io.out.bits.info_data.dir == out_clone.bits.info_data.dir)
        assert(dut.io.out.ready.dir == out_clone.ready.dir)
        assert(dut.io.out.valid.dir == out_clone.valid.dir)
      }
      */
    }
  }
}
