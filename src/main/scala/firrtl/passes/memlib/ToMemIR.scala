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
  */
object ToMemIR extends Pass {
  /** Only annotate memories that are candidates for memory macro replacements
    * i.e. rw, w + r (read, write 1 cycle delay)
    */
  def updateStmts(s: Statement): Statement = s match {
    case m: DefMemory if m.readLatency == 1 && m.writeLatency == 1 &&
        (m.writers.length + m.readwriters.length) == 1 && m.readers.length <= 1 =>
      DefAnnotatedMemory(
        m.info,
        m.name,
        m.dataType,
        m.depth,
        m.writeLatency,
        m.readLatency,
        m.readers,
        m.writers,
        m.readwriters,
        m.readUnderWrite,
        None, // mask granularity annotation
        None  // No reference yet to another memory
      )
    case sx => sx map updateStmts
  }

  def annotateModMems(m: DefModule) = m map updateStmts
  def run(c: Circuit) = c copy (modules = c.modules map annotateModMems)
}
