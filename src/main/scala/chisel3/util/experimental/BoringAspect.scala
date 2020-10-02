// See LICENSE for license details.

package chisel3.util.experimental

import chisel3._
import chisel3.aop.Aspect
import chisel3.aop.injecting.InjectingAspect
import firrtl.AnnotationSeq
import firrtl.options.Dependency
import firrtl.passes.wiring.WiringTransform
import firrtl.stage.RunFirrtlTransformAnnotation

case class BoringAspect[T <: RawModule, M <: RawModule](selectRoots: T => Iterable[M],
                                                        bore: M => (Data, Seq[RawModule], Data, String)) extends Aspect[T] {

  override def toAnnotation(top: T): AnnotationSeq = {
    selectRoots(top).map(bore).flatMap { case (source: Data, through: Seq[RawModule], sink: Data, name: String) =>
      val (_, annos) = BoringUtils.annotateSource(source, name, false, true)
      val sourceAnnos = annos.map(_.toFirrtl)
      var number = 0
      val intermediateAnnos = InjectingAspect((_: RawModule) => through, (m: RawModule) => {
        val intermediate = Wire(chiselTypeOf(source)).suggestName(name)
        intermediate := DontCare
        BoringUtils.addSink(intermediate, name + (if(number == 0) "" else number.toString), false, true)
        number += 1
        BoringUtils.addSource(intermediate, name + number.toString, false, true)
      }).toAnnotation(top)

      val sinkAnnos = BoringUtils.annotateSink(sink, name + number.toString, false, true).map(_.toFirrtl)

      RunFirrtlTransformAnnotation(Dependency[WiringTransform]) +: (sourceAnnos ++ intermediateAnnos ++ sinkAnnos)
    }.toSeq
  }
}

object Example {


}
