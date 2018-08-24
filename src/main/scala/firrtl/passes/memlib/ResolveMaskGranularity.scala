// See LICENSE for license details.

package firrtl.passes
package memlib

import firrtl._
import firrtl.ir._
import firrtl.Utils._
import firrtl.Mappers._
import WrappedExpression.weq
import AnalysisUtils._
import MemTransformUtils._

object AnalysisUtils {
  type Connects = collection.mutable.HashMap[String, Expression]

  /** Builds a map from named component to assigned value
    * Named components are serialized LHS of connections, nodes, invalids
    */
  def getConnects(m: DefModule): Connects = {
    def getConnects(connects: Connects)(s: Statement): Statement = {
      s match {
        case Connect(_, loc, expr) =>
          connects(loc.serialize) = expr
        case DefNode(_, name, value) =>
          connects(name) = value
        case IsInvalid(_, value) =>
          connects(value.serialize) = WInvalid
        case _ => // do nothing
      }
      s map getConnects(connects)
    }
    val connects = new Connects
    m map getConnects(connects)
    connects
  }

  /** Find a connection LHS's origin from a module's list of node-to-node connections
    *   regardless of whether constant propagation has been run.
    * Will search past trivial primop/mux's which do not affect its origin.
    * Limitations: 
    *   - Only works in a module (stops @ module inputs)
    *   - Only does trivial primop/mux's (is not complete)
    * TODO(shunshou): implement more equivalence cases (i.e. a + 0 = a)
    */
  def getOrigin(connects: Connects, s: String): Expression =
    getOrigin(connects)(WRef(s, UnknownType, ExpKind, UNKNOWNGENDER))
  def getOrigin(connects: Connects)(e: Expression): Expression = e match { 
    case Mux(cond, tv, fv, _) =>
      val fvOrigin = getOrigin(connects)(fv)
      val tvOrigin = getOrigin(connects)(tv)
      val condOrigin = getOrigin(connects)(cond)
      if (weq(tvOrigin, one) && weq(fvOrigin, zero)) condOrigin
      else if (weq(condOrigin, one)) tvOrigin
      else if (weq(condOrigin, zero)) fvOrigin
      else if (weq(tvOrigin, fvOrigin)) tvOrigin
      else if (weq(fvOrigin, zero) && weq(condOrigin, tvOrigin)) condOrigin
      else e
    case DoPrim(PrimOps.Or, args, consts, tpe) if args exists (weq(_, one)) => one
    case DoPrim(PrimOps.And, args, consts, tpe) if args exists (weq(_, zero)) => zero
    case DoPrim(PrimOps.Bits, args, Seq(msb, lsb), tpe) =>  
      val extractionWidth = (msb - lsb) + 1
      val nodeWidth = bitWidth(args.head.tpe)
      // if you're extracting the full bitwidth, then keep searching for origin
      if (nodeWidth == extractionWidth) getOrigin(connects)(args.head) else e
    case DoPrim((PrimOps.AsUInt | PrimOps.AsSInt | PrimOps.AsClock), args, _, _) => 
      getOrigin(connects)(args.head)
    // It is a correct optimization to treat ValidIf as a connection
    case ValidIf(cond, value, _) => getOrigin(connects)(value)
    // note: this should stop on a reg, but will stack overflow for combinational loops (not allowed)
    case _: WRef | _: WSubField | _: WSubIndex | _: WSubAccess if kind(e) != RegKind =>
       connects get e.serialize match {
         case Some(ex) => getOrigin(connects)(ex)
         case None => e
       }
    case _ => e
  }
}

/** Determines if a write mask is needed (wmode/en and wmask are equivalent).
  * Populates the maskGran field of DefAnnotatedMemory
  * Annotations:
  *   - maskGran = (dataType size) / (number of mask bits)
  *      - i.e. 1 if bitmask, 8 if bytemask, absent for no mask
  * TODO(shunshou): Add floorplan info?
  */
object ResolveMaskGranularity extends Pass {

  /** Returns the number of mask bits, if used
    */
  def getMaskBits(connects: Connects, wen: Expression, wmask: Expression): Option[Int] = {
    val wenOrigin = getOrigin(connects)(wen)
    val wmaskOrigin = connects.keys filter
      (_ startsWith wmask.serialize) map {s: String => getOrigin(connects, s)}
    // all wmask bits are equal to wmode/wen or all wmask bits = 1(for redundancy checking)
    val redundantMask = wmaskOrigin forall (x => weq(x, wenOrigin) || weq(x, one))
    if (redundantMask) None else Some(wmaskOrigin.size)
  }

  /** Only annotate memories that are candidates for memory macro replacements
    * i.e. rw, w + r (read, write 1 cycle delay)
    */
  def updateStmts(connects: Connects)(s: Statement): Statement = s match {
    case m: DefAnnotatedMemory =>
      val dataBits = bitWidth(m.dataType)
      val rwMasks = m.readwriters map (rw =>
        getMaskBits(connects, memPortField(m, rw, "wmode"), memPortField(m, rw, "wmask")))
      val wMasks = m.writers map (w =>
        getMaskBits(connects, memPortField(m, w, "en"), memPortField(m, w, "mask")))
      val maskGran = (rwMasks ++ wMasks).head match {
        case None =>  None
        case Some(maskBits) => Some(dataBits / maskBits)
      }
      m.copy(maskGran = maskGran)
    case sx => sx map updateStmts(connects)
  }

  def annotateModMems(m: DefModule): DefModule = m map updateStmts(getConnects(m))
  def run(c: Circuit): Circuit = c copy (modules = c.modules map annotateModMems)
}
