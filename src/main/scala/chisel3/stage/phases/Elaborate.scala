// See LICENSE for license details.

package chisel3.stage.phases

import java.io.{PrintWriter, StringWriter}

import chisel3.{ChiselException, Module}
import chisel3.internal.ErrorLog
import chisel3.stage.{ChiselGeneratorAnnotation, ChiselCircuitAnnotation, ChiselOptions}

import firrtl.AnnotationSeq
import firrtl.annotations.DeletedAnnotation
import firrtl.options.{OptionsException, Phase}
import firrtl.options.Viewer.view

object Elaborate extends Phase {

  def transform(annotations: AnnotationSeq): AnnotationSeq = annotations.flatMap {
    case a: ChiselGeneratorAnnotation =>
      try {
        Seq( DeletedAnnotation(name, a), a.elaborate )
      } catch {
        /* todo: How should OptionsExceptions be handled? */
        case e: OptionsException => throw e
        /* todo: What should this return? Should this delete the annotation? */
        case e: ChiselException =>
          val copts = view[ChiselOptions](annotations)
          val stackTrace = if (!copts.printFullStackTrace) {
            e.chiselStackTrace
          } else {
            val s = new StringWriter
            e.printStackTrace(new PrintWriter(s))
            s.toString
          }
          Predef.augmentString(stackTrace).lines.foreach(line => println(s"${ErrorLog.errTag} $line"))
          Seq(DeletedAnnotation(name, a))
      }
    case a => Seq(a)
  }

}
