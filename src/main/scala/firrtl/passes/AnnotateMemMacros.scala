package firrtl.passes

import scala.collection.mutable.{ArrayBuffer,HashMap}
import AnalysisUtils._
import firrtl.WrappedExpression._
import firrtl.ir._
import firrtl._
import firrtl.Utils._

case class AppendableInfo(fields: Map[String,Any]) extends Info {
  def append(a: Map[String,Any]) = this.copy(fields = fields ++ a)
  def append(a: Tuple2[String,Any]): AppendableInfo = append(Map(a))
  def get(f: String) = fields.get(f)
  override def equals(b: Any) = b match {
    case i: AppendableInfo => fields - "info" == i.fields - "info"
    case _ => false
  }
}

object AnalysisUtils {

  def getConnects(m: Module) = {
    val connects = HashMap[String, Expression]()
    def getConnects(s: Statement): Unit = s match {
      case s: Connect  =>
        connects(s.loc.serialize) = s.expr
      case s: PartialConnect =>
        connects(s.loc.serialize) = s.expr
      case s: DefNode =>
        connects(s.name) = s.value
      case s: Block =>
        s.stmts foreach getConnects
      case _ =>
    }
    getConnects(m.body)
    connects.toMap
  }  

  // only works in a module
  def getConnectOrigin(connects: Map[String, Expression], node: String): Expression = 
    if (connects contains node) getConnectOrigin(connects,connects(node))
    else EmptyExpression
  
  // backward searches until PrimOp, Lit or non-trivial Mux appears
  private def getConnectOrigin(connects: Map[String, Expression], e: Expression): Expression = e match {    
    case Mux(cond, tv, fv, _) if we(tv) == we(one) && we(fv) == we(zero) =>
      getConnectOrigin(connects,cond.serialize)
    case _: WRef | _: SubField | _: SubIndex | _: SubAccess if connects contains e.serialize =>
      getConnectOrigin(connects,e.serialize) 
    case _ => e
  }

  def appendInfo[T <: Info](info: T, add: Map[String,Any]) = info match {
    case i: AppendableInfo => i.append(add)
    case _ => AppendableInfo(fields = add + ("info" -> info))
  }
  def appendInfo[T <: Info](info: T, add: Tuple2[String,Any]): AppendableInfo = appendInfo(info,Map(add))
  def getInfo[T <: Info](info: T, k: String) = info match{
    case i: AppendableInfo => i.get(k)
    case _ => None
  }
  def containsInfo[T <: Info](info: T, k: String) = info match {
    case i: AppendableInfo => i.fields.contains(k)
    case _ => false
  }

  def eqMems(a: DefMemory, b: DefMemory) = {
    a.info == b.info &&
    a.dataType == b.dataType &&
    a.depth == b.depth &&
    a.writeLatency == b.writeLatency &&
    a.readLatency == b.readLatency &&
    a.readers == b.readers &&
    a.writers == b.writers &&
    a.readwriters == b.readwriters &&
    a.readUnderWrite == b.readUnderWrite
  }  

}

object AnnotateMemMacros extends Pass {

  def name = "Analyze sequential memories and tag with info for future passes (useMacro,maskGran)"

  def run(c: Circuit) = {

    def annotateModMems(m: Module) = {
      val connects = getConnects(m)

      // returns # of mask bits if used
      def getMaskBits(wen: String, wmask: String): Option[Int] = {
        val wenOrigin = we(getConnectOrigin(connects,wen))
        val one1 = we(one)
        val wmaskOrigin = connects.keys.toSeq.filter(_.startsWith(wmask)).map(x => we(getConnectOrigin(connects,x)))
        // all wmask bits are equal to wmode/wen or all wmask bits = 1(for redundancy checking)
        val redundantMask = wmaskOrigin.map( x => (x == wenOrigin) || (x == one1) ).foldLeft(true)(_ && _)
        if (redundantMask) None else Some(wmaskOrigin.length)
      }

      def updateStmts(s: Statement): Statement = s match {
        // only annotate memories that are candidates for memory macro replacements
        // i.e. rw, w + r (read,write 1 cycle delay)
        case m: DefMemory if m.readLatency == 1 && m.writeLatency == 1 && 
            (m.writers.length + m.readwriters.length) == 1 && m.readers.length <= 1 =>
          val dataBits = bitWidth(m.dataType)
          val rwMasks = m.readwriters map (w => getMaskBits(s"${m.name}.$w.wmode",s"${m.name}.$w.wmask"))
          val wMasks = m.writers map (w => getMaskBits(s"${m.name}.$w.en",s"${m.name}.$w.mask"))
          val maskBits = (rwMasks ++ wMasks).head
          val memAnnotations = Map("useMacro" -> true)
          val tempInfo = appendInfo(m.info,memAnnotations)
          if (maskBits == None) m.copy(info = tempInfo)
          else m.copy(info = tempInfo.append("maskGran" -> dataBits/maskBits.get))       
        case b: Block => Block(b.stmts map updateStmts)
        case s => s
      }

      m.copy(body=updateStmts(m.body))
    } 

    val updatedMods = c.modules map {
      case m: Module => annotateModMems(m)
      case m: ExtModule => m
    }
    c.copy(modules = updatedMods)

  }  

}

// TODO: Add floorplan info?