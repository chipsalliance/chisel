// SPDX-License-Identifier: Apache-2.0

package firrtl
package transforms

import firrtl.annotations._
import firrtl.passes.PassException

/** Indicate that DCE should not be run */
case object NoDCEAnnotation extends NoTargetAnnotation

/** Lets an annotation mark its ReferenceTarget members as DontTouch
  *
  * This permits a transform to run and remove its associated annotations,
  * thus making their ReferenceTargets new candidates for optimization. This
  * removes the need for the pass writer to reason about pre-existing
  * DontTouchAnnotations that may touch the same node.
  */
trait HasDontTouches { self: Annotation =>
  def dontTouches: Iterable[ReferenceTarget]
}

/**
  * A globalized form of HasDontTouches which applies to all ReferenceTargets
  * provided with the annotation
  */
trait DontTouchAllTargets extends HasDontTouches { self: Annotation =>
  def dontTouches: Iterable[ReferenceTarget] = getTargets.collect {
    case rT: ReferenceTarget => rT
  }
}

/** A component that should be preserved
  *
  * DCE treats the component as a top-level sink of the circuit
  */
case class DontTouchAnnotation(target: ReferenceTarget)
    extends SingleTargetAnnotation[ReferenceTarget]
    with DontTouchAllTargets {
  def targets = Seq(target)
  def duplicate(n: ReferenceTarget) = this.copy(n)
}

object DontTouchAnnotation {
  class DontTouchNotFoundException(module: String, component: String)
      extends PassException(
        s"""|Target marked dontTouch ($module.$component) not found!
            |It was probably accidentally deleted. Please check that your custom transforms are not responsible and then
            |file an issue on GitHub: https://github.com/freechipsproject/firrtl/issues/new""".stripMargin
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
case class OptimizableExtModuleAnnotation(target: ModuleName) extends SingleTargetAnnotation[ModuleName] {
  def duplicate(n: ModuleName) = this.copy(n)
}
