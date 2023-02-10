// SPDX-License-Identifier: Apache-2.0

package firrtl
package passes

import firrtl.ir._
import Utils.throwInternalError

/** Remove [[firrtl.ir.ValidIf ValidIf]] and replace [[firrtl.ir.IsInvalid IsInvalid]] with a connection to zero */
object RemoveValidIf {

  /** Returns an [[firrtl.ir.Expression Expression]] equal to zero for a given [[firrtl.ir.GroundType GroundType]]
    * @note Accepts [[firrtl.ir.Type Type]] but dyanmically expects [[firrtl.ir.GroundType GroundType]]
    */
  def getGroundZero(tpe: Type): Expression = tpe match {
    case g: GroundType => Utils.getGroundZero(g)
    case other => throwInternalError(s"Unexpected type $other")
  }

}
