// See LICENSE for license details.

package firrtl
package passes
package memlib
import ir._
import annotations._
import wiring._

class CreateMemoryAnnotations(reader: Option[YamlFileReader]) extends Transform {
  def inputForm = MidForm
  def outputForm = MidForm
  def execute(state: CircuitState): CircuitState = reader match {
    case None => state
    case Some(r) =>
      import CustomYAMLProtocol._
      val configs = r.parse[Config]
      val cN = CircuitName(state.circuit.main)
      val oldAnnos = state.annotations
      val (as, pins) = configs.foldLeft((oldAnnos, Seq.empty[String])) { case ((annos, pins), config) =>
        val source = SourceAnnotation(ComponentName(config.source.name, ModuleName(config.source.module, cN)), config.pin.name)
        (annos, pins :+ config.pin.name)
      }
      state.copy(annotations = PinAnnotation(pins.toSeq) +: as)
  }
}
