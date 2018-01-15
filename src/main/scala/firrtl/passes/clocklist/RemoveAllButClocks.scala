// See license file for details

package firrtl.passes
package clocklist

import firrtl._
import firrtl.ir._
import annotations._
import Utils.error
import java.io.{File, CharArrayWriter, PrintWriter, Writer}
import ClockListUtils._
import Utils._
import memlib.AnalysisUtils._
import memlib._
import Mappers._

/** Remove all statements and ports (except instances/whens/blocks) whose
 *  expressions do not relate to ground types.
 */
object RemoveAllButClocks extends Pass {
  def onStmt(s: Statement): Statement = (s map onStmt) match {
    case DefWire(i, n, ClockType) => s
    case DefNode(i, n, value) if value.tpe == ClockType => s
    case Connect(i, l, r) if l.tpe == ClockType => s
    case sx: WDefInstance => sx
    case sx: DefInstance => sx
    case sx: Block => sx
    case sx: Conditionally => sx
    case _ => EmptyStmt
  }
  def onModule(m: DefModule): DefModule = m match {
    case Module(i, n, ps, b) => Module(i, n, ps.filter(_.tpe == ClockType), squashEmpty(onStmt(b)))
    case ExtModule(i, n, ps, dn, p) => ExtModule(i, n, ps.filter(_.tpe == ClockType), dn, p)
  }
  def run(c: Circuit): Circuit = c.copy(modules = c.modules map onModule)
}
