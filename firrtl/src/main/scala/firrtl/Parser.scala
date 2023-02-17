// SPDX-License-Identifier: Apache-2.0

package firrtl

import firrtl.ir._

object Parser {

  sealed abstract class InfoMode

  case object IgnoreInfo extends InfoMode

  case object UseInfo extends InfoMode

  case class GenInfo(filename: String) extends InfoMode

  case class AppendInfo(filename: String) extends InfoMode

}
