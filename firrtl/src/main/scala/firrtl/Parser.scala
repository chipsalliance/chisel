// SPDX-License-Identifier: Apache-2.0

package firrtl

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
object Parser {

  sealed abstract class InfoMode

  case object IgnoreInfo extends InfoMode

  case object UseInfo extends InfoMode

  case class GenInfo(filename: String) extends InfoMode

  case class AppendInfo(filename: String) extends InfoMode

}
