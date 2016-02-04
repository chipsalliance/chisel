// See LICENSE for license details.

package chiselTests

import Chisel._
import org.scalatest._
import org.scalatest.prop._
import Chisel.testers.BasicTester

class ReadCmd extends Bundle {
  val addr = UInt(width = 32)
}

class ReaderExample extends Module {
  val io = new Bundle {
    val in = DeqIO(new ReadCmd)
    val out = EnqIO(new ReadCmd)
  }
}


class DecoupledSpec extends ChiselFlatSpec {
  "ready" should "an input and valid should be an input for deqIO ports" in {
    class ReaderExampleTester extends BasicTester {
      val c = new ReaderExample

      c.io.in.ready.dir should be (OUTPUT)
      c.io.in.valid.dir should be (INPUT)
      c.io.in.bits.addr.dir should be (INPUT)

      c.io.out.ready.dir should be (INPUT)
      c.io.out.valid.dir should be (OUTPUT)
      c.io.out.bits.addr.dir should be (OUTPUT)

      val cloned = c.io.in.cloneType

      cloned.ready.dir should be (OUTPUT)
      cloned.valid.dir should be (INPUT)
      cloned.bits.addr.dir should be (INPUT)

      stop()
    }

    execute{ new ReaderExampleTester }
  }
}
