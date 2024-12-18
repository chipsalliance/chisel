// SPDX-License-Identifier: Apache-2.0

package chiselTests

class Math extends ChiselPropSpec {
  import chisel3.util._

  property("unsignedBitLength is computed correctly") {
    forAll(safeUIntWidth) {
      case (width: Int) =>
        for (offset <- List(-1, 0, 1)) {
          val n = (1 << width) + offset
          if (n >= 0) {
            val d = unsignedBitLength(n)
            val t = if (n == 0) 0 else if (offset < 0) width else width + 1
            d shouldEqual (t)
          }
        }
    }
  }

  property("signedBitLength is computed correctly") {
    forAll(safeUIntWidth) {
      case (width: Int) =>
        for (offset <- List(-1, 0, 1)) {
          for (mult <- List(-1, +1)) {
            val n = ((1 << (width - 1)) + offset) * mult
            val d = signedBitLength(n)
            val t = n match {
              case -2 => 2
              case -1 => 1
              case 0  => 0
              case 1  => 2
              case 2  => 3
              case _ =>
                if (n > 0) {
                  if (offset < 0) width else width + 1
                } else {
                  if (offset > 0) width + 1 else width
                }
            }
            d shouldEqual (t)
          }
        }
    }
  }
}
