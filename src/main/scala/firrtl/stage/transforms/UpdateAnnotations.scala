// See LICENSE for license details.

package firrtl.stage.transforms

import firrtl.{CircuitState, Transform}
import firrtl.options.Translator

class UpdateAnnotations(val underlying: Transform) extends Transform with WrappedTransform
    with Translator[CircuitState, (CircuitState, CircuitState)] {

  override def execute(c: CircuitState): CircuitState = underlying.transform(c)

  def aToB(a: CircuitState): (CircuitState, CircuitState) = (a, a)

  def bToA(b: (CircuitState, CircuitState)): CircuitState = {
    Transform.remapAnnotations(name, b._1, b._2, logger)
  }

  def internalTransform(b: (CircuitState, CircuitState)): (CircuitState, CircuitState) = {
    val result = Transform.runTransform(name, underlying.transform(b._2), logger)
    (b._1, result)
  }
}

object UpdateAnnotations {

  def apply(a: Transform): UpdateAnnotations = new UpdateAnnotations(a)

}
