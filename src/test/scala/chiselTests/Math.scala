// See LICENSE for license details.

package chiselTests

class Math extends ChiselPropSpec {
  import chisel3.util._

  property ("BitsRequired is computed correctly") {
    forAll(safeUIntWidth) { case (width: Int) =>
      for ( offset <- List(-1, 0, 1)) {
        val n = (1 << width) + offset
        if (n > 0) {
          val addIfPow2 = if (isPow2(n)) 1 else 0
          val l2c = log2Ceil(n) + addIfPow2
          val d = UnsignedBitsRequired(n)
          if (n == 1) {
            d shouldEqual 1
          } else {
            d shouldEqual l2c
          }
        }
      }
    }
  }

  property ("WidthRequired is computed correctly") {
    forAll(safeUIntWidth) { case (width: Int) =>
      for ( offset <- List(-1, 0, 1)) {
        val n = (1 << width) + offset
        if (n > 0) {
          val w = UnsignedWidthRequired(n)
          val d = UnsignedBitsRequired(n)
          w.get shouldEqual d
        }
      }
    }
  }
}
