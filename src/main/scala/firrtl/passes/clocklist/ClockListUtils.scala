// See license file for details

package firrtl.passes
package clocklist

import firrtl._
import firrtl.ir._
import annotations._
import Utils.error
import java.io.{File, CharArrayWriter, PrintWriter, Writer}
import wiring.Lineage
import ClockListUtils._
import Utils._
import memlib.AnalysisUtils._
import memlib._
import Mappers._

object ClockListUtils {
  /** Returns a list of clock outputs from instances of external modules
   */
  def getSourceList(moduleMap: Map[String, DefModule])(lin: Lineage): Seq[String] = {
    val s = lin.foldLeft(Seq[String]()){case (sL, (i, l)) =>
      val sLx = getSourceList(moduleMap)(l)
      val sLxx = sLx map (i + "$" + _)
      sL ++ sLxx
    }
    val sourceList = moduleMap(lin.name) match {
      case ExtModule(i, n, ports, dn, p) =>
        val portExps = ports.flatMap{p => create_exps(WRef(p.name, p.tpe, PortKind, to_gender(p.direction)))}
        portExps.filter(e => (e.tpe == ClockType) && (gender(e) == FEMALE)).map(_.serialize)
      case _ => Nil
    }
    val sx = sourceList ++ s
    sx
  }
  /** Returns a map from instance name to its clock origin.
   *  Child instances are not included if they share the same clock as their parent
   */
  def getOrigins(connects: Connects, me: String, moduleMap: Map[String, DefModule])(lin: Lineage): Map[String, String] = {
    val sep = if(me == "") "" else "$"
    // Get origins from all children
    val childrenOrigins = lin.foldLeft(Map[String, String]()){case (o, (i, l)) =>
      o ++ getOrigins(connects, me + sep + i, moduleMap)(l)
    }
    // If I have a clock, get it
    val clockOpt = moduleMap(lin.name) match {
      case Module(i, n, ports, b) => ports.collectFirst { case p if p.name == "clock" => me + sep + "clock" }
      case ExtModule(i, n, ports, dn, p) => None
    }
    // Return new origins with direct children removed, if they match my clock
    clockOpt match {
      case Some(clock) =>
        val myOrigin = getOrigin(connects, clock).serialize
        childrenOrigins.foldLeft(Map(me -> myOrigin)) { case (o, (childInstance, childOrigin)) =>
          val childrenInstances = lin.children.map { case (instance, _) => me + sep + instance }
          // If direct child shares my origin, omit it
          if(childOrigin == myOrigin && childrenInstances.contains(childInstance)) o else o + (childInstance -> childOrigin)
        }
      case None => childrenOrigins
    }
  }
}
