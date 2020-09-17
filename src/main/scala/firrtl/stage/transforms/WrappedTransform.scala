// SPDX-License-Identifier: Apache-2.0

package firrtl.stage.transforms

import firrtl.Transform

import logger.Logger

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

  final override protected val logger = new Logger(trueUnderlying.getClass.getName)

  override def inputForm = underlying.inputForm
  override def outputForm = underlying.outputForm
  override def prerequisites = underlying.prerequisites
  @deprecated(
    "Due to confusion, 'dependents' is being renamed to 'optionalPrerequisiteOf'. Override the latter instead.",
    "FIRRTL 1.3"
  )
  override def dependents = underlying.dependents
  override def optionalPrerequisiteOf = underlying.optionalPrerequisiteOf
  override final def invalidates(b: Transform): Boolean = underlying.invalidates(b)
  override final lazy val name = underlying.name

}
