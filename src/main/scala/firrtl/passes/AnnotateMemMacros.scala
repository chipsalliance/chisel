// See LICENSE for license details.

package firrtl.passes

import firrtl._
import firrtl.ir._
import firrtl.Utils._
import firrtl.Mappers._
import WrappedExpression.weq
import MemPortUtils.memPortField
import AnalysisUtils._

case class AppendableInfo(fields: Map[String, Any]) extends Info {
  def append(a: Map[String, Any]) = this.copy(fields = fields ++ a)
  def append(a: (String, Any)): AppendableInfo = append(Map(a))
  def get(f: String) = fields.get(f)
  override def equals(b: Any) = b match {
    case i: AppendableInfo => fields - "info" == i.fields - "info"
    case _ => false
  }
}

object AnalysisUtils {
  type Connects = collection.mutable.HashMap[String, Expression]
  def getConnects(m: DefModule): Connects = {
    def getConnects(connects: Connects)(s: Statement): Statement = {
      s match {
        case Connect(_, loc, expr) =>
          connects(loc.serialize) = expr
        case DefNode(_, name, value) =>
          connects(name) = value
        case _ => // do nothing
      }
      s map getConnects(connects)
    }
    val connects = new Connects
    m map getConnects(connects)
    connects
  }

  // takes in a list of node-to-node connections in a given module and looks to find the origin of the LHS.
  // if the source is a trivial primop/mux, etc. that has yet to be optimized via constant propagation,
  // the function will try to search backwards past the primop/mux. 
  // use case: compare if two nodes have the same origin
  // limitation: only works in a module (stops @ module inputs)
  // TODO: more thorough (i.e. a + 0 = a)
  def getConnectOrigin(connects: Connects)(node: String): Expression =
    connects get node match {
      case None => EmptyExpression
      case Some(e) => getOrigin(connects, e)
    }
  def getConnectOrigin(connects: Connects, e: Expression): Expression =
    getConnectOrigin(connects)(e.serialize)

  private def getOrigin(connects: Connects, e: Expression): Expression = e match { 
    case Mux(cond, tv, fv, _) =>
      val fvOrigin = getOrigin(connects, fv)
      val tvOrigin = getOrigin(connects, tv)
      val condOrigin = getOrigin(connects, cond)
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
      if (nodeWidth == extractionWidth) getOrigin(connects, args.head) else e
    case DoPrim((PrimOps.AsUInt | PrimOps.AsSInt | PrimOps.AsClock), args, _, _) => 
      getOrigin(connects, args.head)
    // Todo: It's not clear it's ok to call remove validifs before mem passes...
    case ValidIf(cond, value, ClockType) => getOrigin(connects, value)
    // note: this should stop on a reg, but will stack overflow for combinational loops (not allowed)
    case _: WRef | _: WSubField | _: WSubIndex | _: WSubAccess if kind(e) != RegKind =>
       connects get e.serialize match {
         case Some(ex) => getOrigin(connects, ex)
         case None => e
       }
    case _ => e
  }

  def appendInfo[T <: Info](info: T, add: Map[String, Any]) = info match {
    case i: AppendableInfo => i.append(add)
    case _ => AppendableInfo(fields = add + ("info" -> info))
  }
  def appendInfo[T <: Info](info: T, add: (String, Any)): AppendableInfo = appendInfo(info, Map(add))
  def getInfo[T <: Info](info: T, k: String) = info match {
    case i: AppendableInfo => i.get(k)
    case _ => None
  }
  def containsInfo[T <: Info](info: T, k: String) = info match {
    case i: AppendableInfo => i.fields.contains(k)
    case _ => false
  }

  // memories equivalent as long as all fields (except name) are the same
  def eqMems(a: DefMemory, b: DefMemory) = a == b.copy(name = a.name) 
}

object AnnotateMemMacros extends Pass {
  def name = "Analyze sequential memories and tag with info for future passes(useMacro, maskGran)"

  // returns # of mask bits if used
  def getMaskBits(connects: Connects, wen: Expression, wmask: Expression): Option[Int] = {
    val wenOrigin = getConnectOrigin(connects, wen)
    val wmaskOrigin = connects.keys filter
      (_ startsWith wmask.serialize) map getConnectOrigin(connects)
    // all wmask bits are equal to wmode/wen or all wmask bits = 1(for redundancy checking)
    val redundantMask = wmaskOrigin forall (x => weq(x, wenOrigin) || weq(x, one))
    if (redundantMask) None else Some(wmaskOrigin.size)
  }

  def updateStmts(connects: Connects)(s: Statement): Statement = s match {
    // only annotate memories that are candidates for memory macro replacements
    // i.e. rw, w + r (read, write 1 cycle delay)
    case m: DefMemory if m.readLatency == 1 && m.writeLatency == 1 &&
        (m.writers.length + m.readwriters.length) == 1 && m.readers.length <= 1 =>
      val dataBits = bitWidth(m.dataType)
      val rwMasks = m.readwriters map (rw =>
        getMaskBits(connects, memPortField(m, rw, "wmode"), memPortField(m, rw, "wmask")))
      val wMasks = m.writers map (w =>
        getMaskBits(connects, memPortField(m, w, "en"), memPortField(m, w, "mask")))
      val memAnnotations = Map("useMacro" -> true)
      val tempInfo = appendInfo(m.info, memAnnotations)
      (rwMasks ++ wMasks).head match {
        case None =>
          m copy (info = tempInfo)
        case Some(maskBits) =>
          m.copy(info = tempInfo.append("maskGran" -> dataBits / maskBits))
      }
    case s => s map updateStmts(connects)
  }

  def annotateModMems(m: DefModule) = m map updateStmts(getConnects(m))

  def run(c: Circuit) = c copy (modules = (c.modules map annotateModMems))
}

// TODO: Add floorplan info?
