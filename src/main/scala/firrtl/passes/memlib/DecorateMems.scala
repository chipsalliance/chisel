// See LICENSE for license details.

package firrtl
package passes
package memlib

import firrtl.stage.Forms

class CreateMemoryAnnotations(reader: Option[YamlFileReader]) extends Transform with DependencyAPIMigration {

  override def prerequisites = Forms.MidForm
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf = Forms.MidEmitters
  override def invalidates(a: Transform) = false

  def execute(state: CircuitState): CircuitState = reader match {
    case None => state
    case Some(r) =>
      import CustomYAMLProtocol._
      val configs = r.parse[Config]
      val oldAnnos = state.annotations
      val (as, pins) = configs.foldLeft((oldAnnos, Seq.empty[String])) {
        case ((annos, pins), config) =>
          (annos, pins :+ config.pin.name)
      }
      state.copy(annotations = PinAnnotation(pins.toSeq) +: as)
  }
}
