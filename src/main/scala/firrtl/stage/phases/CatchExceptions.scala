// See LICENSE for license details.

package firrtl.stage.phases

import firrtl.{AnnotationSeq, CustomTransformException, FIRRTLException, FirrtlInternalException, FirrtlUserException,
  Utils}
import firrtl.options.{DependencyManagerException, Phase, PhaseException, OptionsException}
import firrtl.passes.{PassException, PassExceptions}

import scala.util.control.ControlThrowable

class CatchExceptions(val underlying: Phase) extends Phase {

  override final val prerequisites = underlying.prerequisites
  override final val dependents = underlying.dependents
  override final def invalidates(a: Phase): Boolean = underlying.invalidates(a)
  override final lazy val name = underlying.name

  override def transform(a: AnnotationSeq): AnnotationSeq = try {
    underlying.transform(a)
  } catch {
    /* Rethrow the exceptions which are expected or due to the runtime environment (out of memory, stack overflow, etc.).
     * Any UNEXPECTED exceptions should be treated as internal errors. */
    case p @ (_: ControlThrowable | _: FIRRTLException | _: OptionsException | _: FirrtlUserException
                | _: FirrtlInternalException | _: PhaseException | _: DependencyManagerException) => throw p
    case CustomTransformException(cause) => throw cause
    case e: Exception => Utils.throwInternalError(exception = Some(e))
  }

}


object CatchExceptions {

  def apply(p: Phase): CatchExceptions = new CatchExceptions(p)

}
