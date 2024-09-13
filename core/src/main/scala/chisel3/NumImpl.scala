// SPDX-License-Identifier: Apache-2.0

package chisel3

import scala.math.BigDecimal.RoundingMode.{HALF_UP, RoundingMode}
import chisel3.experimental.SourceInfo

private[chisel3] trait NumImpl[T <: Data] {
  self: Num[T] =>

  protected def _minImpl(that: T)(implicit sourceInfo: SourceInfo): T =
    Mux(this < that, this.asInstanceOf[T], that)

  protected def _maxImpl(that: T)(implicit sourceInfo: SourceInfo): T =
    Mux(this < that, that, this.asInstanceOf[T])
}

/** NumbObject has a lot of convenience methods for converting between
  * BigInts and Double and BigDecimal
  */
trait NumObject {
  val MaxBitsBigIntToBigDecimal = 108
  val MaxBitsBigIntToDouble = 53

  /**
    * How to create a bigint from a double with a specific binaryPoint
    * @param x           a double value
    * @param binaryPoint a binaryPoint that you would like to use
    * @return
    */
  def toBigInt(x: Double, binaryPoint: Int): BigInt = {
    val multiplier = math.pow(2, binaryPoint)
    val result = BigInt(math.round(x * multiplier))
    result
  }

  /** Create a bigint from a big decimal with a specific binaryPoint (Int)
    *
    * @param x           the BigDecimal value
    * @param binaryPoint the binaryPoint to use
    * @return
    */
  def toBigInt(x: BigDecimal, binaryPoint: Int): BigInt = toBigInt(x, binaryPoint, HALF_UP)

  /** Create a bigint from a big decimal with a specific binaryPoint (Int)
    *
    * @param x           the BigDecimal value
    * @param binaryPoint the binaryPoint to use
    * @param roundingMode the RoundingMode to use
    * @return
    */
  def toBigInt(x: BigDecimal, binaryPoint: Int, roundingMode: RoundingMode): BigInt = {
    val multiplier = math.pow(2, binaryPoint)
    val result = (x * multiplier).setScale(0, roundingMode).toBigInt
    result
  }

  /**
    * converts a bigInt with the given binaryPoint into the double representation
    * @param i           a bigint
    * @param binaryPoint the implied binaryPoint of @i
    * @return
    */
  def toDouble(i: BigInt, binaryPoint: Int): Double = {
    if (i.bitLength >= 54) {
      throw new ChiselException(
        s"BigInt $i with bitlength ${i.bitLength} is too big, precision lost with > $MaxBitsBigIntToDouble bits"
      )
    }
    val multiplier = math.pow(2, binaryPoint)
    val result = i.toDouble / multiplier
    result
  }

  /**
    * converts a bigInt with the given binaryPoint into the BigDecimal representation
    * @param value           a bigint
    * @param binaryPoint the implied binaryPoint of @i
    * @return
    */
  def toBigDecimal(value: BigInt, binaryPoint: Int): BigDecimal = {
    if (value.bitLength > MaxBitsBigIntToBigDecimal) {
      throw new ChiselException(
        s"BigInt $value with bitlength ${value.bitLength} is too big, precision lost with > $MaxBitsBigIntToBigDecimal bits"
      )
    }
    val multiplier = BigDecimal(1.0) / BigDecimal(math.pow(2, binaryPoint))
    val result = BigDecimal(value) * multiplier
    result
  }

}
