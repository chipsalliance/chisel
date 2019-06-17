// See LICENSE for license details.

package chiselTests

import org.scalacheck.Shrink

class Math extends ChiselPropSpec {
  import chisel3.util._
  // Disable shrinking on error.
  implicit val noShrinkListVal = Shrink[List[Int]](_ => Stream.empty)
  implicit val noShrinkInt = Shrink[Int](_ => Stream.empty)

  property ("unsignedBitLength is computed correctly") {
    forAll(safeUIntWidth) { case (width: Int) =>
      for ( offset <- List(-1, 0, 1)) {
        val n = (1 << width) + offset
        if (n > 0) {
          val d = unsignedBitLength(n)
          val t = if (n == 1) 1 else BigInt(n).bitLength
          d shouldEqual (t)
        }
      }
    }
  }

  property ("signedBitLength is computed correctly") {
    forAll(safeUIntWidth) { case (width: Int) =>
      for ( offset <- List(-1, 0, 1)) {
        for ( mult <- List(-1, +1)) {
          val n = ((1 << (width - 1)) + offset) * mult
          val d = signedBitLength(n)
          val t = BigInt(n).bitLength + 1
          d shouldEqual (t)
        }
      }
    }
  }
}
