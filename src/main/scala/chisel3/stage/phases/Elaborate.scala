// See LICENSE for license details.

package chisel3.stage.phases

import java.io.{PrintWriter, StringWriter}

import chisel3.ChiselException
import chisel3.internal.ErrorLog
import chisel3.stage.{ChiselGeneratorAnnotation, ChiselOptions}
import firrtl.AnnotationSeq
import firrtl.options.Viewer.view
import firrtl.options.{OptionsException, Phase}

/** Elaborate all [[chisel3.stage.ChiselGeneratorAnnotation]]s into [[chisel3.stage.ChiselCircuitAnnotation]]s.
  */
class Elaborate extends Phase {

  /**
    * @todo Change this to print to STDERR (`Console.err.println`)
    */
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
