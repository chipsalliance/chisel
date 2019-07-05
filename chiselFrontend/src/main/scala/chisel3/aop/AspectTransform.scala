// See LICENSE for license details.

package chisel3.aop

import chisel3.experimental.{DesignAnnotation, RawModule}
import firrtl._

import scala.collection.mutable

class AspectTransform extends Transform {
  override def inputForm: CircuitForm = ChirrtlForm
  override def outputForm: CircuitForm = ChirrtlForm

  override def execute(state: CircuitState): CircuitState = {

    var dut: Option[RawModule] = None
    val aspects = mutable.ArrayBuffer[Aspect[_]]()

    val remainingAnnotations = state.annotations.flatMap {
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
      state.copy(annotations = remainingAnnotations ++ newAnnotations)
    } else state

  }
}
