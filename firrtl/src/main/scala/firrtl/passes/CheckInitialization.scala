// SPDX-License-Identifier: Apache-2.0

package firrtl.passes

import firrtl._
import firrtl.ir._
import firrtl.Utils._

/** Reports errors for any references that are not fully initialized
  *
  * @note This pass looks for [[firrtl.WVoid]]s left behind by [[ExpandWhens]]
  * @note Assumes single connection (ie. no last connect semantics)
  */
object CheckInitialization {

  class RefNotInitializedException(info: Info, mname: String, name: String, trace: Seq[Statement])
      extends PassException(
        s"$info : [module $mname]  Reference $name is not fully initialized.\n" +
          trace.map(s => s"  ${get_info(s)} : ${s.serialize}").mkString("\n")
      )

}
