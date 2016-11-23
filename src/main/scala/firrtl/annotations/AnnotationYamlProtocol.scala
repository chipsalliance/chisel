// See LICENSE for license details.

package firrtl
package annotations

import net.jcazevedo.moultingyaml._

object AnnotationYamlProtocol extends DefaultYamlProtocol {
  // bottom depends on top
  implicit object AnnotationYamlFormat extends YamlFormat[Annotation] {
    def write(a: Annotation) =
      YamlArray(
        YamlString(a.targetString),
        YamlString(a.transformClass),
        YamlString(a.value))

    def read(value: YamlValue) = {
      value.asYamlObject.getFields(
        YamlString("targetString"),
        YamlString("transformClass"),
        YamlString("value")) match {
        case Seq(
          YamlString(targetString),
          YamlString(transformClass),
          YamlString(value)) =>
          new Annotation(toTarget(targetString), Class.forName(transformClass).asInstanceOf[Class[_ <: Transform]], value)
        case _ => deserializationError("Color expected")
      }
    }
    def toTarget(string: String) = string.split('.').toSeq match {
      case Seq(c) => CircuitName(c)
      case Seq(c, m) => ModuleName(m, CircuitName(c))
      case Nil => error("BAD")
      case s => ComponentName(s.drop(2).mkString("."), ModuleName(s(1), CircuitName(s(0))))
    }
  }
}
