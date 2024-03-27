// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.testers.BasicTester


class BoolSpec extends ChiselFlatSpec with Utils {

  "implication" should "work in RTL" in {
    assertTesterPasses {
      new BasicTester {
        chisel3.assert((false.B |-> false.B) === true.B,  "0 -> 0 = 1")
        chisel3.assert((false.B |-> true.B)  === true.B,  "0 -> 1 = 1")
        chisel3.assert((true.B  |-> false.B) === false.B, "1 -> 0 = 0")
        chisel3.assert((true.B  |-> true.B)  === true.B,  "1 -> 1 = 1")
        stop()
      }
    }
  }
}
