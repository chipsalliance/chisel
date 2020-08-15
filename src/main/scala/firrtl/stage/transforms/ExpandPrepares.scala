// See LICENSE for license details.

package firrtl.stage.transforms

import firrtl.{CircuitState, Transform}

class ExpandPrepares(val underlying: Transform) extends Transform with WrappedTransform {

  /* Assert that this is not wrapping other transforms. */
  underlying match {
    case _: WrappedTransform =>
      throw new Exception(
        s"'ExpandPrepares' must not wrap other 'WrappedTransforms', but wraps '${underlying.getClass.getName}'"
      )
    case _ =>
  }

  override def execute(c: CircuitState): CircuitState = {
    underlying.transform(underlying.prepare(c))
  }

}

object ExpandPrepares {

  def apply(a: Transform): ExpandPrepares = new ExpandPrepares(a)

}
