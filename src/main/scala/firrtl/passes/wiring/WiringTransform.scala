// See LICENSE for license details.

package firrtl.passes
package wiring

import firrtl._
import firrtl.ir._
import firrtl.Utils._
import firrtl.Mappers._
import scala.collection.mutable
import firrtl.annotations._
import WiringUtils._

/** A component, e.g. register etc. Must be declared only once under the TopAnnotation
  */
object SourceAnnotation {
  def apply(target: ComponentName, pin: String): Annotation = Annotation(target, classOf[WiringTransform], s"source $pin")

  private val matcher = "source (.+)".r
  def unapply(a: Annotation): Option[(ComponentName, String)] = a match {
    case Annotation(ComponentName(n, m), _, matcher(pin)) => Some((ComponentName(n, m), pin))
    case _ => None
  }
}

/** A module, e.g. ExtModule etc., that should add the input pin
  */
object SinkAnnotation {
  def apply(target: ModuleName, pin: String): Annotation = Annotation(target, classOf[WiringTransform], s"sink $pin")

  private val matcher = "sink (.+)".r
  def unapply(a: Annotation): Option[(ModuleName, String)] = a match {
    case Annotation(ModuleName(n, c), _, matcher(pin)) => Some((ModuleName(n, c), pin))
    case _ => None
  }
}

/** A module under which all sink module must be declared, and there is only
  * one source component
  */
object TopAnnotation {
  def apply(target: ModuleName, pin: String): Annotation = Annotation(target, classOf[WiringTransform], s"top $pin")

  private val matcher = "top (.+)".r
  def unapply(a: Annotation): Option[(ModuleName, String)] = a match {
    case Annotation(ModuleName(n, c), _, matcher(pin)) => Some((ModuleName(n, c), pin))
    case _ => None
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
class WiringTransform extends Transform with SimpleRun {
  def inputForm = MidForm
  def outputForm = MidForm
  def passSeq(wis: Seq[WiringInfo]) =
    Seq(new Wiring(wis),
        InferTypes,
        ResolveKinds,
        ResolveGenders)
  def execute(state: CircuitState): CircuitState = getMyAnnotations(state) match {
    case Nil => CircuitState(state.circuit, state.form)
    case p => 
      val sinks = mutable.HashMap[String, Set[String]]()
      val sources = mutable.HashMap[String, String]()
      val tops = mutable.HashMap[String, String]()
      val comp = mutable.HashMap[String, String]()
      p.foreach { 
        case SinkAnnotation(m, pin) =>
          sinks(pin) = sinks.getOrElse(pin, Set.empty) + m.name
        case SourceAnnotation(c, pin) =>
          sources(pin) = c.module.name
          comp(pin) = c.name
        case TopAnnotation(m, pin) => tops(pin) = m.name
      }
      (sources.size, tops.size, sinks.size, comp.size) match {
        case (0, 0, p, 0) => state.copy(annotations = None)
        case (s, t, p, c) if (p > 0) & (s == t) & (t == c) =>
          val wis = tops.foldLeft(Seq[WiringInfo]()) { case (seq, (pin, top)) =>
            seq :+ WiringInfo(sources(pin), comp(pin), sinks("pin:" + pin), pin, top)
          }
          state.copy(circuit = runPasses(state.circuit, passSeq(wis)), annotations = None)
        case _ => error("Wrong number of sources, tops, or sinks!")
      }
  }
}
