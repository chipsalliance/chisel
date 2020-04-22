// See LICENSE for license details.

package firrtl.stage.transforms

import firrtl.Transform

/** A [[firrtl.Transform]] that "wraps" a second [[firrtl.Transform Transform]] to do some work before and after the
  * second [[firrtl.Transform Transform]].
  *
  * This is intended to synergize with the [[firrtl.options.DependencyManager.wrappers]] method.
  * @see [[firrtl.stage.transforms.CatchCustomTransformExceptions]]
  * @see [[firrtl.stage.transforms.TrackTransforms]]
  * @see [[firrtl.stage.transforms.UpdateAnnotations]]
  */
trait WrappedTransform { this: Transform =>

  /** The underlying [[firrtl.Transform]] */
  val underlying: Transform

  /** Return the original [[firrtl.Transform]] if this wrapper is wrapping other wrappers. */
  lazy final val trueUnderlying: Transform = underlying match {
    case a: WrappedTransform => a.trueUnderlying
    case _ => underlying
  }

  override final def inputForm = underlying.inputForm
  override final def outputForm = underlying.outputForm
  override final def prerequisites = underlying.prerequisites
  override final def optionalPrerequisites = underlying.optionalPrerequisites
  override final def dependents = underlying.dependents
  override final def invalidates(b: Transform): Boolean = underlying.invalidates(b)
  override final lazy val name = underlying.name

}
