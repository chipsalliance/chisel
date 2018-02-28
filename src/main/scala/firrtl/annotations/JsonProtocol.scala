
package firrtl
package annotations

import scala.util.Try

import org.json4s._
import org.json4s.native.JsonMethods._
import org.json4s.native.Serialization
import org.json4s.native.Serialization.{read, write, writePretty}

import firrtl.ir._
import firrtl.Utils.error

object JsonProtocol {

  // Helper for error messages
  private def prettifyJsonInput(in: JsonInput): String = {
    def defaultToString(base: String, obj: Any): String = s"$base@${obj.hashCode.toHexString}"
    in match {
      case FileInput(file) => file.toString
      case StringInput(o) => defaultToString("String", o)
      case ReaderInput(o) => defaultToString("Reader", o)
      case StreamInput(o) => defaultToString("Stream", o)
    }
  }

  class TransformClassSerializer extends CustomSerializer[Class[_ <: Transform]](format => (
    { case JString(s) => Class.forName(s).asInstanceOf[Class[_ <: Transform]] },
    { case x: Class[_] => JString(x.getName) }
  ))
  // TODO Reduce boilerplate?
  class NamedSerializer extends CustomSerializer[Named](format => (
    { case JString(s) => AnnotationUtils.toNamed(s) },
    { case named: Named => JString(named.serialize) }
  ))
  class CircuitNameSerializer extends CustomSerializer[CircuitName](format => (
    { case JString(s) => AnnotationUtils.toNamed(s).asInstanceOf[CircuitName] },
    { case named: CircuitName => JString(named.serialize) }
  ))
  class ModuleNameSerializer extends CustomSerializer[ModuleName](format => (
    { case JString(s) => AnnotationUtils.toNamed(s).asInstanceOf[ModuleName] },
    { case named: ModuleName => JString(named.serialize) }
  ))
  class ComponentNameSerializer extends CustomSerializer[ComponentName](format => (
    { case JString(s) => AnnotationUtils.toNamed(s).asInstanceOf[ComponentName] },
    { case named: ComponentName => JString(named.serialize) }
  ))

  /** Construct Json formatter for annotations */
  def jsonFormat(tags: Seq[Class[_ <: Annotation]]) = {
    Serialization.formats(FullTypeHints(tags.toList)).withTypeHintFieldName("class") +
      new TransformClassSerializer + new NamedSerializer + new CircuitNameSerializer +
      new ModuleNameSerializer + new ComponentNameSerializer
  }

  /** Serialize annotations to a String for emission */
  def serialize(annos: Seq[Annotation]): String = serializeTry(annos).get

  def serializeTry(annos: Seq[Annotation]): Try[String] = {
    val tags = annos.map(_.getClass).distinct
    implicit val formats = jsonFormat(tags)
    Try(writePretty(annos))
  }

  def deserialize(in: JsonInput): Seq[Annotation] = deserializeTry(in).get

  def deserializeTry(in: JsonInput): Try[Seq[Annotation]] = Try({
    def throwError() = throw new InvalidAnnotationFileException(prettifyJsonInput(in))
    val parsed = parse(in)
    val annos = parsed match {
      case JArray(objs) => objs
      case _ => throwError()
    }
    // Gather classes so we can deserialize arbitrary Annotations
    val classes = annos.map({
      case JObject(("class", JString(c)) :: tail) => c
      case _ => throwError()
    }).distinct
    val loaded = classes.map(Class.forName(_).asInstanceOf[Class[_ <: Annotation]])
    implicit val formats = jsonFormat(loaded)
    read[List[Annotation]](in)
  })
}
