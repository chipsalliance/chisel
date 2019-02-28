package chisel3.libs.initialize

import chisel3.Data
import chisel3.core.RunFirrtlTransform
import chisel3.experimental.ChiselAnnotation
import firrtl.{CircuitState, HighForm, LowForm, RenameMap, Transform}
import firrtl.annotations.{Annotation, BrittleAnnotation, Component}
import firrtl.ir.Expression


//object Initialize {
//  def apply(c: Component, reset: Component, value: Data): ChiselAnnotation = {
//
//  }
//}

case class Initialize(c: Component, reset: Component, refs: Seq[Component], value: Expression) extends RunFirrtlTransform with BrittleAnnotation {
  def targets = Seq(c, reset) ++ refs

  def duplicate(targets: Seq[Component]): BrittleAnnotation = {
    Initialize(targets.head, targets(1), targets.drop(2), value)
  }

  override def transformClass: Class[_ <: Transform] = classOf[InitializeTransform]

  override def toFirrtl: Annotation = this
}

class InitializeTransform extends firrtl.Transform {
  val inputForm = HighForm
  val outputForm = HighForm
  override def execute(state: CircuitState): CircuitState = {
    state
  }
}
