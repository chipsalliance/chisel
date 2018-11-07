
package firrtl
package annotations

import scala.util.{Try, Failure}

import org.json4s._
import org.json4s.native.JsonMethods._
import org.json4s.native.Serialization
import org.json4s.native.Serialization.{read, write, writePretty}

import firrtl.ir._
import firrtl.Utils.error

object JsonProtocol {
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
  class TransformSerializer extends CustomSerializer[Transform](format => (
    { case JString(s) =>
      try {
        Class.forName(s).asInstanceOf[Class[_ <: Transform]].newInstance()
      } catch {
        case e: java.lang.InstantiationException => throw new FIRRTLException(
          "NoSuchMethodException during construction of serialized Transform. Is your Transform an inner class?", e)
        case t: Throwable => throw t
      }},
    { case x: Transform => JString(x.getClass.getName) }
  ))


  class TargetSerializer extends CustomSerializer[Target](format => (
    { case JString(s) => Target.deserialize(s) },
    { case named: Target => JString(named.serialize) }
  ))
  class GenericTargetSerializer extends CustomSerializer[GenericTarget](format => (
    { case JString(s) => Target.deserialize(s).asInstanceOf[GenericTarget] },
    { case named: GenericTarget => JString(named.serialize) }
  ))
  class CircuitTargetSerializer extends CustomSerializer[CircuitTarget](format => (
    { case JString(s) => Target.deserialize(s).asInstanceOf[CircuitTarget] },
    { case named: CircuitTarget => JString(named.serialize) }
  ))
  class ModuleTargetSerializer extends CustomSerializer[ModuleTarget](format => (
    { case JString(s) => Target.deserialize(s).asInstanceOf[ModuleTarget] },
    { case named: ModuleTarget => JString(named.serialize) }
  ))
  class InstanceTargetSerializer extends CustomSerializer[InstanceTarget](format => (
    { case JString(s) => Target.deserialize(s).asInstanceOf[InstanceTarget] },
    { case named: InstanceTarget => JString(named.serialize) }
  ))
  class ReferenceTargetSerializer extends CustomSerializer[ReferenceTarget](format => (
    { case JString(s) => Target.deserialize(s).asInstanceOf[ReferenceTarget] },
    { case named: ReferenceTarget => JString(named.serialize) }
  ))

  /** Construct Json formatter for annotations */
  def jsonFormat(tags: Seq[Class[_ <: Annotation]]) = {
    Serialization.formats(FullTypeHints(tags.toList)).withTypeHintFieldName("class") +
      new TransformClassSerializer + new NamedSerializer + new CircuitNameSerializer +
      new ModuleNameSerializer + new ComponentNameSerializer + new TargetSerializer +
      new GenericTargetSerializer + new CircuitTargetSerializer + new ModuleTargetSerializer +
      new InstanceTargetSerializer + new ReferenceTargetSerializer + new TransformSerializer
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
    val parsed = parse(in)
    val annos = parsed match {
      case JArray(objs) => objs
      case x => throw new InvalidAnnotationJSONException(
        s"Annotations must be serialized as a JArray, got ${x.getClass.getSimpleName} instead!")
    }
    // Gather classes so we can deserialize arbitrary Annotations
    val classes = annos.map({
      case JObject(("class", JString(c)) :: tail) => c
      case obj => throw new InvalidAnnotationJSONException(s"Expected field 'class' not found! $obj")
    }).distinct
    val loaded = classes.map(Class.forName(_).asInstanceOf[Class[_ <: Annotation]])
    implicit val formats = jsonFormat(loaded)
    read[List[Annotation]](in)
  }).recoverWith {
    // Translate some generic errors to specific ones
    case e: java.lang.ClassNotFoundException =>
      Failure(new AnnotationClassNotFoundException(e.getMessage))
    case e: org.json4s.ParserUtil.ParseException =>
      Failure(new InvalidAnnotationJSONException(e.getMessage))
  }.recoverWith { // If the input is a file, wrap in InvalidAnnotationFileException
    case e => in match {
      case FileInput(file) =>
        Failure(new InvalidAnnotationFileException(file, e))
      case _ => Failure(e)
    }
  }
}
