package chisel3.aop

import chisel3.core.RawModule
import firrtl._

import scala.collection.mutable

/** Transform associated with a Concern
  *
  * Consumes the [[DesignAnnotation]] and converts every [[Concern]] into their annotations prior to execute
  */
trait ConcernTransform extends Transform {
  override def inputForm: CircuitForm = ChirrtlForm
  override def outputForm: CircuitForm = ChirrtlForm

  override def prepare(state: CircuitState): CircuitState = {

    var dut: Option[RawModule] = None
    val concerns = mutable.ArrayBuffer[Concern[_, _]]()

    val remainingAnnotations = state.annotations.flatMap {
      case DesignAnnotation(d) =>
        dut = Some(d)
        Nil
      case a: Concern[_, _] =>
        concerns += a
        Nil
      case other => Seq(other)
    }
    if(dut.isDefined) {
      val newAnnotations = concerns.flatMap { _.resolveAspects(dut.get) }
      state.copy(annotations = remainingAnnotations ++ newAnnotations)
    } else state

  }
}
