package chisel3.aop

import chisel3.core.{DesignAnnotation, RawModule}
import firrtl._

import scala.collection.mutable

/** Transform associated with a Concern
  *
  * Consumes the [[DesignAnnotation]] and converts every [[Aspect]] into their annotations prior to execute
  */
class AspectTransform extends Transform {
  override def inputForm: CircuitForm = ChirrtlForm
  override def outputForm: CircuitForm = ChirrtlForm

  override def execute(state: CircuitState): CircuitState = {

    var dut: Option[RawModule] = None
    val aspects = mutable.ArrayBuffer[Aspect[_, _]]()

    val remainingAnnotations = state.annotations.flatMap {
      case DesignAnnotation(d) =>
        dut = Some(d)
        Nil
      case a: Aspect[_, _] =>
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
