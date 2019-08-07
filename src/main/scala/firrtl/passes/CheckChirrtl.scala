// See LICENSE for license details.

package firrtl.passes

import firrtl.ir._

object CheckChirrtl extends Pass with CheckHighFormLike {
  def errorOnChirrtl(info: Info, mname: String, s: Statement): Option[PassException] = None
}
