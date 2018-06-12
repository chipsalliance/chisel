// See LICENSE for license details.

package chisel3.util

import chisel3.core.RunFirrtlTransform
import chisel3.experimental.ChiselAnnotation
import chisel3.internal.InstanceId
import firrtl.{CircuitForm, CircuitState, LowForm, TargetDirAnnotation, Transform}
import firrtl.annotations.{Named, SingleTargetAnnotation}

case class ChiselLoadMemoryAnnotation(target: InstanceId, fileName: String)
  extends ChiselAnnotation
    with RunFirrtlTransform {

  def transformClass : Class[LoadMemoryTransform] = classOf[LoadMemoryTransform]

  def toFirrtl: LoadMemoryAnnotation = LoadMemoryAnnotation(target.toNamed, fileName)
}

case class LoadMemoryAnnotation(target: Named, value1: String) extends SingleTargetAnnotation[Named] {
  def duplicate(n: Named): LoadMemoryAnnotation = this.copy(target = n)
}

//noinspection ScalaStyle
class LoadMemoryTransform extends Transform {
  def inputForm  : CircuitForm = LowForm
  def outputForm : CircuitForm = LowForm

  def execute(state: CircuitState): CircuitState = {
    val targetDir = state.annotations.collectFirst { case td: TargetDirAnnotation => td }
    println(s"target dir is ${targetDir.getOrElse("no dir")}")
    val processedAnnotations = state.annotations.map {
      case LoadMemoryAnnotation(t, value) => LoadMemoryAnnotation(t, value + ":seen")
      case other => other
    }
    println(s"got here")
    state.copy(annotations = processedAnnotations)
  }
}
