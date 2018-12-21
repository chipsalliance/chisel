// See LICENSE for license details.

package firrtl.transforms

import firrtl.{CircuitForm, CircuitState, Transform}

/** Transform that applies an identity function. This returns an unmodified [[CircuitState]].
  * @param form the input and output [[CircuitForm]]
  */
class IdentityTransform(form: CircuitForm) extends Transform {

  final override def inputForm: CircuitForm = form
  final override def outputForm: CircuitForm = form

  final def execute(state: CircuitState): CircuitState = state

}
