// See LICENSE for license details.

package firrtl.passes
package memlib

import firrtl.Mappers._
import firrtl.ir._

/** Annotates sequential memories that are candidates for macro replacement.
  * Requirements for macro replacement:
  *   - read latency and write latency of one
  *   - only one readwrite port or write port
  *   - zero or one read port
  *   - undefined read-under-write behavior
  */
object ToMemIR extends Pass {
  /** Only annotate memories that are candidates for memory macro replacements
    * i.e. rw, w + r (read, write 1 cycle delay) and read-under-write "undefined."
    */
  import ReadUnderWrite._
  def updateStmts(s: Statement): Statement = s match {
    case m @ DefMemory(_,_,_,_,1,1,r,w,rw,Undefined) if (w.length + rw.length) == 1 && r.length <= 1 =>
      DefAnnotatedMemory(m)
    case sx => sx map updateStmts
  }

  def annotateModMems(m: DefModule) = m map updateStmts
  def run(c: Circuit) = c copy (modules = c.modules map annotateModMems)
}
