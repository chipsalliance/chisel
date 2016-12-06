// See LICENSE for license details.

package chiselTests

import chisel3._

class WidthSpec extends ChiselFlatSpec {
  "Literals without specified widths" should "get the minimum legal width" in {
    "hdeadbeef".U.getWidth should be (32)
    "h_dead_beef".U.getWidth should be (32)
    "h0a".U.getWidth should be (4)
    "h1a".U.getWidth should be (5)
    "h0".U.getWidth should be (1)
    1.U.getWidth should be (1)
    1.S.getWidth should be (2)
  }
}
