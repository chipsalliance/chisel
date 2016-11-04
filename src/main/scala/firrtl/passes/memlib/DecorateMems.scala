// See LICENSE for license details.

package firrtl
package passes
package memlib
import ir._
import Annotations._
import wiring._

class CreateMemoryAnnotations(reader: Option[YamlFileReader]) extends Transform {
  def inputForm = MidForm
  def outputForm = MidForm
  override def name = "Create Memory Annotations"
  def execute(state: CircuitState): CircuitState = reader match {
    case None => state
    case Some(r) =>
      import CustomYAMLProtocol._
      r.parse[Config] match {
        case Seq(config) =>
          val cN = CircuitName(state.circuit.main)
          val top = TopAnnotation(ModuleName(config.top.name, cN))
          val source = SourceAnnotation(ComponentName(config.source.name, ModuleName(config.source.module, cN)))
          val pin = PinAnnotation(cN, config.pin.name)
          state.copy(annotations = Some(AnnotationMap(Seq(top, source, pin))))
        case Nil => state
        case _ => error("Can only have one config in yaml file")
      }
  }
}
