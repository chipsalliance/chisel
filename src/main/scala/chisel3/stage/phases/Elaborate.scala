// See LICENSE for license details.

package chisel3.stage.phases

import java.io.{PrintWriter, StringWriter}

import chisel3.{ChiselException, Module}
import chisel3.internal.ErrorLog
import chisel3.stage.{ChiselGeneratorAnnotation, ChiselCircuitAnnotation, ChiselOptions}

import firrtl.AnnotationSeq
import firrtl.options.{OptionsException, Phase}
import firrtl.options.Viewer.view

object Elaborate extends Phase {

  def transform(annotations: AnnotationSeq): AnnotationSeq = annotations.flatMap {
    case a: ChiselGeneratorAnnotation =>
      try {
        Some(a.elaborate)
      } catch {
        case e: OptionsException => throw e
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
          Some(a)
      }
    case a => Some(a)
  }

}
