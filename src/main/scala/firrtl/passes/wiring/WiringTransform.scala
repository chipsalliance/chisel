package firrtl.passes
package wiring

import firrtl._
import firrtl.ir._
import firrtl.Utils._
import firrtl.Mappers._
import scala.collection.mutable
import firrtl.Annotations._
import WiringUtils._

/** A component, e.g. register etc. Must be declared only once under the TopAnnotation
  */
case class SourceAnnotation(target: ComponentName, tID: TransID) extends Annotation with Loose with Unstable {
  def duplicate(n: Named) = n match {
    case n: ComponentName => this.copy(target = n)
    case _ => throwInternalError
  }
}

/** A module, e.g. ExtModule etc., that should add the input pin
  */
case class SinkAnnotation(target: ModuleName, tID: TransID, pin: String) extends Annotation with Loose with Unstable {
  def duplicate(n: Named) = n match {
    case n: ModuleName => this.copy(target = n)
    case _ => throwInternalError
  }
}

/** A module under which all sink module must be declared, and there is only
  * one source component
  */
case class TopAnnotation(target: ModuleName, tID: TransID) extends Annotation with Loose with Unstable {
  def duplicate(n: Named) = n match {
    case n: ModuleName => this.copy(target = n)
    case _ => throwInternalError
  }
}

/** Add pins to modules and wires a signal to them, under the scope of a specified top module
  * Description:
  *   Adds a pin to each sink module
  *   Punches ports up from the source signal to the specified top module
  *   Punches ports down to each sink module
  *   Wires the source up and down, connecting to all sink modules
  * Restrictions:
  *   - Can only have one source module instance under scope of the specified top
  *   - All instances of each sink module must be under the scope of the specified top
  * Notes:
  *   - No module uniquification occurs (due to imposed restrictions)
  */
class WiringTransform(transID: TransID) extends Transform with SimpleRun {
  def passSeq(wi: WiringInfo) =
    Seq(new Wiring(wi),
        InferTypes,
        ResolveKinds,
        ResolveGenders)
  def execute(c: Circuit, map: AnnotationMap) = map get transID match {
    case Some(p) => 
      val sinks = mutable.HashMap[String, String]()
      val sources = mutable.Set[String]()
      val tops = mutable.Set[String]()
      val comp = mutable.Set[String]()
      p.values.foreach { a =>
        a match {
          case SinkAnnotation(m, _, pin) => sinks(m.name) = pin
          case SourceAnnotation(c, _) =>
            sources += c.module.name
            comp += c.name
          case TopAnnotation(m, _) => tops += m.name
        }
      }
      (sources.size, tops.size, sinks.size, comp.size) match {
        case (0, 0, p, 0) => TransformResult(c)
        case (1, 1, p, 1) if p > 0 => run(c, passSeq(WiringInfo(sources.head, comp.head, sinks.toMap, tops.head)))
        case _ => error("Wrong number of sources, tops, or sinks!")
      }
    case None => TransformResult(c)
  }
}
