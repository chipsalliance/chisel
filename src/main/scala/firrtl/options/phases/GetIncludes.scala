// See LICENSE for license details.

package firrtl.options.phases

import net.jcazevedo.moultingyaml._

import firrtl.AnnotationSeq
import firrtl.annotations.{AnnotationFileNotFoundException, JsonProtocol, LegacyAnnotation}
import firrtl.annotations.AnnotationYamlProtocol._
import firrtl.options.{InputAnnotationFileAnnotation, Phase, StageUtils}

import java.io.File

import scala.collection.mutable
import scala.util.{Try, Failure}

/** Recursively expand all [[InputAnnotationFileAnnotation]]s in an [[AnnotationSeq]] */
class GetIncludes extends Phase {

  /** Read all [[annotations.Annotation]] from a file in JSON or YAML format
    * @param filename a JSON or YAML file of [[annotations.Annotation]]
    * @throws annotations.AnnotationFileNotFoundException if the file does not exist
    */
  private def readAnnotationsFromFile(filename: String): AnnotationSeq = {
    val file = new File(filename).getCanonicalFile
    if (!file.exists) { throw new AnnotationFileNotFoundException(file) }
    JsonProtocol.deserializeTry(file).recoverWith { case jsonException =>
      // Try old protocol if new one fails
      Try {
        val yaml = io.Source.fromFile(file).getLines().mkString("\n").parseYaml
        val result = yaml.convertTo[List[LegacyAnnotation]]
        val msg = s"$file is a YAML file!\n" + (" "*9) + "YAML Annotation files are deprecated! Use JSON"
        StageUtils.dramaticWarning(msg)
        result
      }.orElse { // Propagate original JsonProtocol exception if YAML also fails
        Failure(jsonException)
      }
    }.get
  }

  /** Recursively read all [[Annotation]]s from any [[InputAnnotationFileAnnotation]]s while making sure that each file is
    * only read once
    * @param includeGuard filenames that have already been read
    * @param annos a sequence of annotations
    * @return the original annotation sequence with any discovered annotations added
    */
  private def getIncludes(includeGuard: mutable.Set[String] = mutable.Set())
                         (annos: AnnotationSeq): AnnotationSeq = {
    annos.flatMap {
      case a @ InputAnnotationFileAnnotation(value) =>
        if (includeGuard.contains(value)) {
          StageUtils.dramaticWarning(s"Annotation file ($value) already included! (Did you include it more than once?)")
          None
        } else {
          includeGuard += value
          getIncludes(includeGuard)(readAnnotationsFromFile(value))
        }
      case x => Seq(x)
    }
  }

  def transform(annotations: AnnotationSeq): AnnotationSeq = getIncludes()(annotations)

}
