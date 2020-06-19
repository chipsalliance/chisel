// See LICENSE for license details.

package firrtl.passes

import firrtl.Transform
import firrtl.ir._
import firrtl.options.Dependency

object CheckChirrtl extends Pass with CheckHighFormLike {

  override val optionalPrerequisiteOf = firrtl.stage.Forms.ChirrtlForm ++
    Seq( Dependency(CInferTypes),
         Dependency(CInferMDir),
         Dependency(RemoveCHIRRTL) )

  override def invalidates(a: Transform) = false

  def errorOnChirrtl(info: Info, mname: String, s: Statement): Option[PassException] = None
}
