// See LICENSE for license details.

package examples

import scala.collection.mutable

/* Useful utilities for testing vending machines */
object VendingMachineUtils {
  abstract class Coin(val name: String, val value: Int)
  // US Coins
  case object Penny extends Coin("penny", 1)
  case object Nickel extends Coin("nickel", 5)
  case object Dime extends Coin("dime", 10)
  case object Quarter extends Coin("quarter", 25)

  // Harry Potter Coins
  case object Knut extends Coin("knut", Penny.value * 2) // Assuming 1 Knut is worth $0.02
  case object Sickle extends Coin("sickle", Knut.value * 29)
  case object Galleon extends Coin("galleon", Sickle.value * 17)

  def getExpectedResults(inputs: Seq[Option[Coin]], sodaCost: Int): Seq[Boolean] = {
    var value = 0
    val outputs = mutable.ArrayBuffer.empty[Boolean]
    for (input <- inputs) {
      val incValue = input match {
        case Some(coin) => coin.value
        case None => 0
      }
      if (value >= sodaCost) {
        outputs.append(true)
        value = 0
      } else {
        outputs.append(false)
        value += incValue
      }
    }
    outputs
  }
}
