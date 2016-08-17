// See LICENSE for license details.

package chiselTests

import chisel3.core.FixedPoint
import org.scalatest.{Matchers, FlatSpec}

class FixedPointSpec extends FlatSpec with Matchers {
  behavior of "fixed point utilities"

  they should "allow conversion between doubles and the bigints needed to represent them" in {
    val initialDouble = 0.125
    val bigInt = FixedPoint.toBigInt(initialDouble, 4)
    val finalDouble = FixedPoint.toDouble(bigInt, 4)

    initialDouble should be (finalDouble)
  }
}
