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

/** A class for all exceptions originating from firrtl.passes.wiring */
case class WiringException(msg: String) extends PassException(msg)

/** An extractor of annotated source components */
object SourceAnnotation {
  def apply(target: ComponentName, pin: String): Annotation =
    Annotation(target, classOf[WiringTransform], s"source $pin")

  private val matcher = "source (.+)".r
  def unapply(a: Annotation): Option[(ComponentName, String)] = a match {
    case Annotation(ComponentName(n, m), _, matcher(pin)) =>
      Some((ComponentName(n, m), pin))
    case _ => None
  }
}

/** An extractor of annotation sink components or modules */
object SinkAnnotation {
  def apply(target: Named, pin: String): Annotation =
    Annotation(target, classOf[WiringTransform], s"sink $pin")

  private val matcher = "sink (.+)".r
  def unapply(a: Annotation): Option[(Named, String)] = a match {
    case Annotation(ModuleName(n, c), _, matcher(pin)) =>
      Some((ModuleName(n, c), pin))
    case Annotation(ComponentName(n, m), _, matcher(pin)) =>
      Some((ComponentName(n, m), pin))
    case _ => None
  }
}

/** Wires a Module's Source Component to one or more Sink
  * Modules/Components
  *
  * Sinks are wired to their closest source through their lowest
  * common ancestor (LCA). Verbosely, this modifies the circuit in
  * the following ways:
  *   - Adds a pin to each sink module
  *   - Punches ports up from source signals to the LCA
  *   - Punches ports down from LCAs to each sink module
  *   - Wires sources up to LCA, sinks down from LCA, and across each LCA
  *
  * @throws WiringException if a sink is equidistant to two sources
  */
class WiringTransform extends Transform {
  def inputForm: CircuitForm = MidForm
  def outputForm: CircuitForm = HighForm

  /** Defines the sequence of Transform that should be applied */
  private def transforms(w: Seq[WiringInfo]): Seq[Transform] = Seq(
    new Wiring(w),
    ToWorkingIR
  )

  def execute(state: CircuitState): CircuitState = getMyAnnotations(state) match {
    case Nil => state
    case p =>
      val sinks = mutable.HashMap[String, Seq[Named]]()
      val sources = mutable.HashMap[String, ComponentName]()
      p.foreach {
        case SinkAnnotation(m, pin) =>
          sinks(pin) = sinks.getOrElse(pin, Seq.empty) :+ m
        case SourceAnnotation(c, pin) =>
          sources(pin) = c
      }
      (sources.size, sinks.size) match {
        case (0, p) => state
        case (s, p) if (p > 0) =>
          val wis = sources.foldLeft(Seq[WiringInfo]()) { case (seq, (pin, source)) =>
            seq :+ WiringInfo(source, sinks(pin), pin)
          }
          transforms(wis).foldLeft(state) { (in, xform) => xform.runTransform(in) }
        case _ => error("Wrong number of sources or sinks!")
      }
  }
}
