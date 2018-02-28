// See LICENSE for license details.

package firrtl
package annotations

import net.jcazevedo.moultingyaml._

object AnnotationYamlProtocol extends DefaultYamlProtocol {
  // bottom depends on top
  implicit object AnnotationYamlFormat extends YamlFormat[LegacyAnnotation] {
    def write(a: LegacyAnnotation) = YamlObject(
      YamlString("targetString") -> YamlString(a.targetString),
      YamlString("transformClass") -> YamlString(a.transformClass),
      YamlString("value") -> YamlString(a.value)
    )

    def read(yamlValue: YamlValue): LegacyAnnotation = {
      try {
        yamlValue.asYamlObject.getFields(
          YamlString("targetString"),
          YamlString("transformClass"),
          YamlString("value")) match {
          case Seq(YamlString(targetString), YamlString(transformClass), YamlString(value)) =>
            LegacyAnnotation(toTarget(targetString),
                             Class.forName(transformClass).asInstanceOf[Class[_ <: Transform]],
                             value)
          case _ => deserializationError("LegacyAnnotation expected")
        }
      }
      catch {
        case annotationException: AnnotationException =>
          Utils.error(
            s"Error: ${annotationException.getMessage} while parsing annotation from yaml\n${yamlValue.prettyPrint}")
        case annotationException: FIRRTLException =>
          Utils.error(
            s"Error: ${annotationException.getMessage} while parsing annotation from yaml\n${yamlValue.prettyPrint}")
      }
    }
    def toTarget(string: String): Named = string.split("""\.""", -1).toSeq match {
      case Seq(c) => CircuitName(c)
      case Seq(c, m) => ModuleName(m, CircuitName(c))
      case Nil => Utils.error("BAD")
      case s =>
        val componentString = s.drop(2).mkString(".")
        ComponentName(componentString, ModuleName(s.tail.head, CircuitName(s.head)))
    }
  }
}
