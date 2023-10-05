// SPDX-License-Identifier: Apache-2.0

package chisel3.stage.phases

import chisel3.aop.Aspect
import chisel3.RawModule
import chisel3.stage.DesignAnnotation
import firrtl.AnnotationSeq
import firrtl.options.Phase

import scala.collection.mutable

/** Phase that consumes all Aspects and calls their toAnnotationSeq methods.
  *
  * Consumes the [[chisel3.stage.DesignAnnotation]] and converts every `Aspect` into their annotations prior to executing FIRRTL
  */
class AspectPhase extends Phase {
  def transform(annotations: AnnotationSeq): AnnotationSeq = {
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
    if (dut.isDefined) {
      val newAnnotations = aspects.flatMap { _.resolveAspect(dut.get, remainingAnnotations) }
      remainingAnnotations ++ newAnnotations
    } else annotations
  }
}
