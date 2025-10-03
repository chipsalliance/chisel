// SPDX-License-Identifier: Apache-2.0

package firrtl
package annotations

import firrtl.ir._
import firrtl.stage.AllowUnrecognizedAnnotations
import logger.LazyLogging

import scala.util.{Failure, Success, Try}

import org.json4s._
import org.json4s.native.JsonMethods._
import org.json4s.native.Serialization
import org.json4s.native.Serialization.{read, write, writePretty}

import scala.collection.mutable
import java.io.{StringWriter, Writer}

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
trait HasSerializationHints {
  // For serialization of complicated constructor arguments, let the annotation
  // writer specify additional type hints for relevant classes that might be
  // contained within
  def typeHints: Seq[Class[_]]
}

/** Similar to [[HasSerializationHints]] but for types whose serialization classes
  * need to be overridden
  */
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
trait HasSerializationOverrides {
  // For serialization of complicated constructor arguments, let the annotation
  // writer specify additional type hints for relevant classes that might be
  // contained within
  def typeOverrides: Seq[(Class[_], String)]
}

/** Mix this in to override what class name is used for serialization
  *
  * Note that this breaks automatic deserialization.
  */
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
trait OverrideSerializationClass {
  def serializationClassOverride: String
}

/** Wrapper [[Annotation]] for Annotations that cannot be serialized */
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class UnserializeableAnnotation(error: String, content: String) extends NoTargetAnnotation

/** Custom type hints for FIRRTL annotations that respect user override of the class */
private case class FirrtlAnnotationTypeHints(hints: List[Class[_]], hintOverride: Map[Class[_], String])
    extends TypeHints {
  private val underlying = FullTypeHints(hints)
  override val typeHintFieldName = "class"
  // For some reason, json4s requires both directions to serialize
  private val classFromHintOverride: Map[String, Class[_]] = hintOverride.map(_.swap)
  def hintFor(clazz: Class[_]): Option[String] = {
    hintOverride.get(clazz).orElse(underlying.hintFor(clazz))
  }
  def classFor(hint: String, parent: Class[_]): Option[Class[_]] = {
    classFromHintOverride.get(hint).orElse(underlying.classFor(hint, parent))
  }
}

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
object JsonProtocol extends LazyLogging {
  private val GetClassPattern = "[^']*'([^']+)'.*".r

  // TODO Reduce boilerplate?
  class NamedSerializer
      extends CustomSerializer[Named](format =>
        (
          { case JString(s) => Target.deserialize(s).toNamed },
          { case named: Named => JString(named.toTarget.serialize) }
        )
      )
  class ModuleNameSerializer
      extends CustomSerializer[ModuleName](format =>
        (
          { case JString(s) => Target.deserialize(s).toNamed.asInstanceOf[ModuleName] },
          { case named: ModuleName => JString(named.toTarget.serialize) }
        )
      )
  class ComponentNameSerializer
      extends CustomSerializer[ComponentName](format =>
        (
          { case JString(s) => Target.deserialize(s).toNamed.asInstanceOf[ComponentName] },
          { case named: ComponentName => JString(named.toTarget.serialize) }
        )
      )
  class LoadMemoryFileTypeSerializer
      extends CustomSerializer[MemoryLoadFileType](format =>
        (
          { case JString(s) => MemoryLoadFileType.deserialize(s) },
          { case named: MemoryLoadFileType => JString(named.serialize) }
        )
      )

  class TargetSerializer
      extends CustomSerializer[Target](format =>
        (
          { case JString(s) => Target.deserialize(s) },
          { case named: Target => JString(named.serialize) }
        )
      )
  class GenericTargetSerializer
      extends CustomSerializer[GenericTarget](format =>
        (
          { case JString(s) => Target.deserialize(s).asInstanceOf[GenericTarget] },
          { case named: GenericTarget => JString(named.serialize) }
        )
      )
  class ModuleTargetSerializer
      extends CustomSerializer[ModuleTarget](format =>
        (
          { case JString(s) => Target.deserialize(s).asInstanceOf[ModuleTarget] },
          { case named: ModuleTarget => JString(named.serialize) }
        )
      )
  class InstanceTargetSerializer
      extends CustomSerializer[InstanceTarget](format =>
        (
          { case JString(s) => Target.deserialize(s).asInstanceOf[InstanceTarget] },
          { case named: InstanceTarget => JString(named.serialize) }
        )
      )
  class ReferenceTargetSerializer
      extends CustomSerializer[ReferenceTarget](format =>
        (
          { case JString(s) => Target.deserialize(s).asInstanceOf[ReferenceTarget] },
          { case named: ReferenceTarget => JString(named.serialize) }
        )
      )
  class IsModuleSerializer
      extends CustomSerializer[IsModule](format =>
        (
          { case JString(s) => Target.deserialize(s).asInstanceOf[IsModule] },
          { case named: IsModule => JString(named.serialize) }
        )
      )
  class IsMemberSerializer
      extends CustomSerializer[IsMember](format =>
        (
          { case JString(s) => Target.deserialize(s).asInstanceOf[IsMember] },
          { case named: IsMember => JString(named.serialize) }
        )
      )
  class CompleteTargetSerializer
      extends CustomSerializer[CompleteTarget](format =>
        (
          { case JString(s) => Target.deserialize(s).asInstanceOf[CompleteTarget] },
          { case named: CompleteTarget => JString(named.serialize) }
        )
      )

  class UnrecognizedAnnotationSerializer
      extends CustomSerializer[JObject](format =>
        (
          { case JObject(s) => JObject(s) },
          { case UnrecognizedAnnotation(underlying) => underlying }
        )
      )

  /** Construct Json formatter for annotations */
  def jsonFormat(tags: Seq[Class[_]]): Formats = jsonFormat(tags, Map())

  /** Construct Json formatter for annotations */
  def jsonFormat(tags: Seq[Class[_]], hintOverride: Map[Class[_], String]): Formats = {
    Serialization.formats(FirrtlAnnotationTypeHints(tags.toList, hintOverride)) +
      new NamedSerializer +
      new ModuleNameSerializer + new ComponentNameSerializer + new TargetSerializer +
      new GenericTargetSerializer + new ModuleTargetSerializer +
      new InstanceTargetSerializer + new ReferenceTargetSerializer +
      new LoadMemoryFileTypeSerializer + new IsModuleSerializer + new IsMemberSerializer +
      new CompleteTargetSerializer + new UnrecognizedAnnotationSerializer
  }

  /** Serialize annotations to a String for emission */
  def serialize(annos: Seq[Annotation]): String = serializeTry(annos).get

  private def findUnserializeableAnnos(
    annos: Seq[Annotation]
  )(
    implicit formats: Formats
  ): Seq[(Annotation, Throwable)] =
    annos.map(a => a -> Try(write(a))).collect { case (a, Failure(e)) => (a, e) }

  private def getTagsAndOverrides(annos: Seq[Annotation]): (List[Class[_]], Map[Class[_], String]) = {
    val tags = mutable.ListBuffer.empty[Class[_]]
    val seenTags = mutable.Set.empty[Class[_]]
    def addTag(clazz: Class[_]): Unit = {
      if (!seenTags(clazz)) {
        tags += clazz
        seenTags += clazz
      }
    }

    val tagOverride = mutable.Map.empty[Class[_], String]

    for (anno <- annos) {
      addTag(anno.getClass)
      anno match {
        case anno: HasSerializationHints =>
          anno.typeHints.foreach(addTag(_))
        case _ => ()
      }
      anno match {
        case anno: HasSerializationOverrides =>
          anno.typeOverrides.foreach { case (clazz, name) =>
            val existing = tagOverride.put(clazz, name)
            if (existing.isDefined && existing.get != name) {
              throw new Exception(
                s"Class $clazz has multiple serialization class overrides: ${existing.get}, $name"
              )
            }
          }
        case _ => ()
      }
      anno match {
        case anno: OverrideSerializationClass =>
          val existing = tagOverride.put(anno.getClass, anno.serializationClassOverride)
          if (existing.isDefined && existing.get != anno.serializationClassOverride) {
            throw new Exception(
              s"Class ${anno.getClass.getName} has multiple serialization class overrides: ${existing.get}, ${anno.serializationClassOverride}"
            )
          }
        case _ => ()
      }
    }
    (tags.toList, tagOverride.toMap)
  }

  def serializeTry(annos: Seq[Annotation]): Try[String] = serializeTry(annos, new StringWriter).map(_.toString)

  /** Serialize annotations to a [[java.io.Writer]]
    *
    * @param annos Annotations to serialize
    * @param out Writer to which the serialized annotations will be written
    * @return
    */
  def serializeTry[W <: Writer](annos: Iterable[Annotation], out: W): Try[W] = {
    val (tags, hintOverride) = getTagsAndOverrides(annos.toSeq)

    implicit val formats = jsonFormat(tags, hintOverride)
    Try(writePretty(annos, out)).recoverWith { case e: org.json4s.MappingException =>
      val badAnnos = findUnserializeableAnnos(annos.toSeq)
      Failure(if (badAnnos.isEmpty) e else UnserializableAnnotationException(badAnnos))
    }
  }

  /** Serialize annotations to JSON while wrapping unserializeable ones with [[UnserializeableAnnotation]]
    *
    * @note this is slower than standard serialization
    */
  def serializeRecover(annos: Seq[Annotation]): String = {
    val (autoTags, hintOverride) = getTagsAndOverrides(annos)

    val tags = classOf[UnserializeableAnnotation] +: autoTags
    implicit val formats = jsonFormat(tags, hintOverride)

    val safeAnnos = annos.map { anno =>
      Try(write(anno)) match {
        case Success(_) => anno
        case Failure(e) => UnserializeableAnnotation(e.getMessage, anno.toString)
      }
    }
    writePretty(safeAnnos)
  }

  /** Deserialize JSON input into a Seq[Annotation]
    *
    * @param in JsonInput, can be file or string
    * @param allowUnrecognizedAnnotations is set to true if command line contains flag to allow this behavior
    * @return
    */
  def deserialize(in: JsonInput, allowUnrecognizedAnnotations: Boolean = false): Seq[Annotation] = {
    deserializeTry(in, allowUnrecognizedAnnotations).get
  }

  def deserializeTry(in: JsonInput, allowUnrecognizedAnnotations: Boolean = false): Try[Seq[Annotation]] = Try {
    val parsed: JValue = parse(in)
    val annos = parsed match {
      case JArray(objs) => objs
      case x =>
        throw new InvalidAnnotationJSONException(
          s"Annotations must be serialized as a JArray, got ${x.getClass.getName} instead!"
        )
    }

    /* Tries to extract class name from the mapping exception */
    def getAnnotationNameFromMappingException(mappingException: MappingException): String = {
      mappingException.getMessage match {
        case GetClassPattern(name) => name
        case other                 => other
      }
    }

    // Recursively gather typeHints by pulling the "class" field from JObjects
    // Json4s should emit this as the first field in all serialized classes
    // Setting requireClassField mandates that all JObjects must provide a typeHint,
    // this used on the first invocation to check all annotations do so
    def findTypeHints(classInst: Seq[JValue], requireClassField: Boolean = false): Seq[String] = classInst
      .flatMap({
        case JObject(fields) =>
          val hint = fields.collectFirst { case ("class", JString(name)) => name }
          if (requireClassField && hint.isEmpty)
            throw new InvalidAnnotationJSONException(s"Expected field 'class' not found! $fields")
          hint ++: findTypeHints(fields.map(_._2))
        case JArray(arr) => findTypeHints(arr)
        case _           => Seq()
      })
      .distinct

    // I don't much like this var here, but it has made it much simpler
    // to maintain backward compatibility with the exception test structure
    var classNotFoundBuildingLoaded = false
    val classes = findTypeHints(annos, true)
    val loaded = classes.flatMap { x =>
      (try {
        Some(Class.forName(x))
      } catch {
        case _: java.lang.ClassNotFoundException =>
          classNotFoundBuildingLoaded = true
          None
      }): Option[Class[_]]
    }
    implicit val formats = jsonFormat(loaded)
    try {
      read[List[Annotation]](in)
    } catch {
      case e: org.json4s.MappingException =>
        // If we get here, the build `read` failed to process an annotation
        // So we will map the annos one a time, wrapping the JSON of the unrecognized annotations
        val exceptionList = new mutable.ArrayBuffer[String]()
        val firrtlAnnos = annos.map { jsonAnno =>
          try {
            jsonAnno.extract[Annotation]
          } catch {
            case mappingException: org.json4s.MappingException =>
              exceptionList += getAnnotationNameFromMappingException(mappingException)
              UnrecognizedAnnotation(jsonAnno)
          }
        }

        if (firrtlAnnos.contains(AllowUnrecognizedAnnotations) || allowUnrecognizedAnnotations) {
          firrtlAnnos
        } else {
          logger.error(
            "Annotation parsing found unrecognized annotations\n" +
              "This error can be ignored with an AllowUnrecognizedAnnotationsAnnotation" +
              " or command line flag --allow-unrecognized-annotations\n" +
              exceptionList.mkString("\n")
          )
          if (classNotFoundBuildingLoaded) {
            val distinctProblems = exceptionList.distinct
            val problems = distinctProblems.take(10).mkString(", ")
            val dots = if (distinctProblems.length > 10) {
              ", ..."
            } else {
              ""
            }
            throw UnrecogizedAnnotationsException(s"($problems$dots)")
          } else {
            throw e
          } // throw the mapping exception
        }
    }
  }.recoverWith {
    // Translate some generic errors to specific ones
    case e: java.lang.ClassNotFoundException =>
      Failure(AnnotationClassNotFoundException(e.getMessage))
    // Eat the stack traces of json4s exceptions
    case e @ (_: org.json4s.ParserUtil.ParseException | _: org.json4s.MappingException) =>
      Failure(InvalidAnnotationJSONException(e.getMessage))
  }.recoverWith { // If the input is a file, wrap in InvalidAnnotationFileException
    case e: UnrecogizedAnnotationsException =>
      in match {
        case FileInput(file) =>
          Failure(InvalidAnnotationFileException(file, e))
        case _ =>
          Failure(e)
      }
    case e: FirrtlUserException =>
      in match {
        case FileInput(file) =>
          Failure(InvalidAnnotationFileException(file, e))
        case _ => Failure(e)
      }
  }
}
