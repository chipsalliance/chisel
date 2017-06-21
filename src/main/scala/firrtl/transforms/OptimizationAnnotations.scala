
package firrtl
package transforms

import firrtl.annotations._
import firrtl.passes.PassException

/** Indicate that DCE should not be run */
object NoDCEAnnotation {
  val marker = "noDCE!"
  val transform = classOf[DeadCodeElimination]
  def apply(): Annotation = Annotation(CircuitTopName, transform, marker)
  def unapply(a: Annotation): Boolean = a match {
    case Annotation(_, targetXform, value) if targetXform == transform && value == marker => true
    case _ => false
  }
}

/** A component that should be preserved
  *
  * DCE treats the component as a top-level sink of the circuit
  */
object DontTouchAnnotation {
  private val marker = "DONTtouch!"
  def apply(target: ComponentName): Annotation = Annotation(target, classOf[Transform], marker)

  def unapply(a: Annotation): Option[ComponentName] = a match {
    case Annotation(component: ComponentName, _, value) if value == marker => Some(component)
    case _ => None
  }

  class DontTouchNotFoundException(module: String, component: String) extends PassException(
    s"Component marked DONT Touch ($module.$component) not found!\n" +
    "Perhaps it is an aggregate type? Currently only leaf components are supported.\n" +
    "Otherwise it was probably accidentally deleted. Please check that your custom passes are not" +
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
object OptimizableExtModuleAnnotation {
  private val marker = "optimizableExtModule!"
  def apply(target: ModuleName): Annotation = Annotation(target, classOf[Transform], marker)

  def unapply(a: Annotation): Option[ModuleName] = a match {
    case Annotation(component: ModuleName, _, value) if value == marker => Some(component)
    case _ => None
  }
}
