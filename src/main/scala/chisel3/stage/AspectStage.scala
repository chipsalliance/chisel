// See LICENSE for license details.

package chisel3.stage

import chisel3.aop.Aspect
import chisel3.experimental.{DesignAnnotation, RawModule}
import firrtl.AnnotationSeq
import firrtl.options.{Shell, Stage}

import scala.collection.mutable

/** Stage associated with all Aspects
  *
  * Consumes the [[chisel3.experimental.DesignAnnotation]] and converts every [[Aspect]] into their annotations prior to executing FIRRTL
  */
class AspectStage extends Stage {
  val shell: Shell = new Shell("aspect") //TODO: This?

  def run(annotations: AnnotationSeq): AnnotationSeq = {
    var dut: Option[RawModule] = None
    val aspects = mutable.ArrayBuffer[Aspect[_]]()

    val remainingAnnotations = annotations.flatMap {
      case DesignAnnotation(d) =>
        dut = Some(d)
        Nil
      case a: Aspect[_] =>
        aspects += a
        Nil
      case other => Seq(other)
    }
    if(dut.isDefined) {
      val newAnnotations = aspects.flatMap { _.resolveAspect(dut.get) }
      remainingAnnotations ++ newAnnotations
    } else annotations
  }
}

