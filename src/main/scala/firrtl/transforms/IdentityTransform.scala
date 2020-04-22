// See LICENSE for license details.

package firrtl.transforms

import firrtl.{CircuitForm, CircuitState, Transform}

/** Transform that applies an identity function. This returns an unmodified [[CircuitState]].
  * @param form the input and output [[CircuitForm]]
  */
@deprecated(
  "mix-in firrtl.options.IdentityLike[CircuitState]. IdentityTransform will be removed in 1.4.",
  "FIRRTL 1.3"
)
class IdentityTransform(form: CircuitForm) extends Transform {

  final override def inputForm: CircuitForm = form

  final override def outputForm: CircuitForm = form

  final def execute(state: CircuitState): CircuitState = state

}
