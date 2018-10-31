
package firrtl
package transforms

import firrtl.annotations._
import firrtl.passes.PassException
import firrtl.transforms

/** Indicate that DCE should not be run */
case object NoDCEAnnotation extends NoTargetAnnotation

/** A component that should be preserved
  *
  * DCE treats the component as a top-level sink of the circuit
  */
case class DontTouchAnnotation(target: ReferenceTarget) extends SingleTargetAnnotation[ReferenceTarget] {
  def targets = Seq(target)
  def duplicate(n: ReferenceTarget) = this.copy(n)
}

object DontTouchAnnotation {
  class DontTouchNotFoundException(module: String, component: String) extends PassException(
    s"Target marked dontTouch ($module.$component) not found!\n" +
    "It was probably accidentally deleted. Please check that your custom transforms are not" +
    "responsible and then file an issue on Github."
  )

  def errorNotFound(module: String, component: String) =
    throw new DontTouchNotFoundException(module, component)
}

/** An [[firrtl.ir.ExtModule]] that can be optimized
  *
  * Firrtl does not know the semantics of an external module. This annotation provides some
  * "greybox" information that the external module does not have any side effects. In particular,
  * this means that the external module can be Dead Code Eliminated.
  *
  * @note Unlike [[DontTouchAnnotation]], we don't care if the annotation is deleted
  */
case class OptimizableExtModuleAnnotation(target: ModuleName) extends
    SingleTargetAnnotation[ModuleName] {
  def duplicate(n: ModuleName) = this.copy(n)
}
